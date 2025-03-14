# Define import
import os
import re
import time
import numpy as np
import evaluate
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from torch.nn import MSELoss
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from transformers import DataCollatorWithPadding
from src.config import MAX_LENGTH, LOG_DIR, DEFAULT_MODEL, BATCH_SIZE
from src.utils import (
    task_to_fields,
    task_to_num_labels,
    task_to_metric,
    num_processes,
    load_glue_dataset_from_dir,
)


def sample_dataset(dataset, task_name, train_sample=2000, val_sample=8, all=False):
    if "test" in dataset:
        del dataset["test"]
    if all:
        return dataset

    train_sample = min(train_sample, len(dataset["train"]))
    val_sample = min(val_sample, len(dataset["validation"]))

    if task_name == "mnli":
        # Handle both validation sets for MNLI
        val_sample = min(val_sample, len(dataset["validation_matched"]))
        dataset["train"] = dataset["train"].shuffle().select(range(train_sample))
        dataset["validation_matched"] = (
            dataset["validation_matched"].shuffle().select(range(val_sample))
        )
        dataset["validation_mismatched"] = (
            dataset["validation_mismatched"].shuffle().select(range(val_sample))
        )
        del dataset["test_matched"]
        del dataset["test_mismatched"]
        return dataset

    # For other tasks
    dataset["train"] = dataset["train"].shuffle().select(range(train_sample))
    dataset["validation"] = dataset["validation"].shuffle().select(range(val_sample))

    return dataset


def tokenize_function(examples, tokenizer, task_name, max_length=MAX_LENGTH):
    fields = task_to_fields.get(task_name, None)

    if not fields:
        raise ValueError(
            f"Task {task_name} not found in task_to_fields or gen_fields"
        )

    if len(fields) == 1:
        # sst2 case
        return tokenizer(
            examples[fields[0]], truncation=True, max_length=max_length, padding=False
        )
    else:
        # the rest hopefully
        return tokenizer(
            examples[fields[0]],
            examples[fields[1]],
            truncation=True,
            max_length=max_length,
            padding=False,
        )


def preprocess_dataset(dataset, tokenizer, task_name):
    dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer, task_name),
        batched=True,
        num_proc=num_processes,
    )
    return dataset


def compute_loss_for_task(model, inputs, task_name):
    if task_name != "stsb":
        return model(**inputs).loss
    # MSE loss for STS-B
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = MSELoss()
    loss = loss_fct(logits.view(-1), labels.view(-1))
    return loss


class CustomTrainer(Trainer):
    def __init__(self, *args, task_name=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_name = task_name

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = compute_loss_for_task(model, inputs, self.task_name)
        return (loss, model(**inputs)) if return_outputs else loss


def setup_trainer(model, dataset, tokenizer, data_collator, model_name, task_name):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"/tmp/{model_name}",
        overwrite_output_dir=True,
        eval_strategy="no",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        save_strategy="no",
        dataloader_num_workers=num_processes,
        learning_rate=5e-5,
        num_train_epochs=3,
        weight_decay=3e-5,
        logging_dir=f"{LOG_DIR}/tensorboard_{model_name}",
        logging_steps=100,
        report_to="tensorboard",
        fp16=True,
        seed=np.random.randint(1e6),
    )

    # Define Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=(
            dataset["validation"]
            if task_name != "mnli"
            else dataset["validation_matched"]
        ),
        tokenizer=tokenizer,
        data_collator=data_collator,
        task_name=task_name,
    )
    return trainer


def compute_metrics(trainer, dataset, task_name, batch_size=256):
    def get_predictions_and_labels(trainer, dataset, batch_size):
        predictions = []
        labels = []
        for i in range(0, len(dataset), batch_size):
            batch = dataset.select(range(i, min(i + batch_size, len(dataset))))
            outputs = trainer.predict(batch)
            preds = outputs.predictions
            if isinstance(preds, tuple):
                preds = preds[0]
            predictions.append(preds)
            labels.append(outputs.label_ids)
        return np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0)

    if task_name == "mnli":
        matched_preds, matched_labels = get_predictions_and_labels(trainer, dataset["validation_matched"], batch_size)
        mismatched_preds, mismatched_labels = get_predictions_and_labels(trainer, dataset["validation_mismatched"], batch_size)

        metric = evaluate.load(task_to_metric[task_name][0])
        matched_result = metric.compute(predictions=np.argmax(matched_preds, axis=1), references=matched_labels)
        mismatched_result = metric.compute(predictions=np.argmax(mismatched_preds, axis=1), references=mismatched_labels)

        return {
            f"{task_name}_matched": matched_result,
            f"{task_name}_mismatched": mismatched_result,
        }

    # Regular compute metrics for other tasks
    preds, labels = get_predictions_and_labels(trainer, dataset["validation"], batch_size)

    metric = evaluate.load(task_to_metric[task_name][0])
    if task_name == "stsb":
        result = metric.compute(predictions=preds, references=labels)
    else:
        result = metric.compute(predictions=np.argmax(preds, axis=1), references=labels)

    return {task_name: result}


def get_model_name(model_path):
    model_name = re.sub(r"/checkpoint-.*", "", model_path)
    model_name = model_name.split("/")[-1]
    print(f"Model name: {model_name}")
    return model_name


def fine_tune_on_all_tasks(
    num_iterations: int,
    model_path: str,
    is_phonetic: bool = False,
    all=False,
):
    start = time.time()
    results = defaultdict(lambda: defaultdict(list))
    model_name = get_model_name(model_path)

    def _fine_tune_on_all_tasks(current_iteration: int = 1):
        one_iter_res = defaultdict(dict)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

        for task_name, num_labels in task_to_num_labels.items():
            try:
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_path, num_labels=num_labels, 
                    ignore_mismatched_sizes=True,
                    output_hidden_states=False,
                )
            except EnvironmentError as e:
                print(f"Error loading model: {e}")
                continue

            model.train()
            print(f"################Training {model_name} on {task_name} with {num_labels}", end=" ")
            print(f"labels on iteration {current_iteration}/{num_iterations}################")
            print(f"Loading dataset {task_name} ...")
            dataset = load_glue_dataset_from_dir(task_name, is_phonetic)
            print("Sampling dataset ...")
            dataset = sample_dataset(dataset, task_name, all=all)
            print("Tokenizing dataset ...")
            dataset = preprocess_dataset(dataset, tokenizer, task_name)
            trainer = setup_trainer(
                model, dataset, tokenizer, data_collator, model_name, task_name
            )
            print("Training model ...")
            trainer.train()
            print("Evaluating model ...")
            model.eval()
            task_result = compute_metrics(trainer, dataset, task_name)
            one_iter_res.update(task_result)
        return one_iter_res

    futures = [
        _fine_tune_on_all_tasks(current_iteration=i + 1)
        for i in range(num_iterations)
    ]

    keys = futures[0].keys()
    for result in futures:
        for key in keys:
            for metric in result[key].keys():
                results[key][metric].append(result[key][metric])

    # Compute mean and std of results
    for key in keys:
        for metric in results[key].keys():
            results[key][metric] = {
                "mean": np.mean(results[key][metric]),
                "std": np.std(results[key][metric]),
            }

    file_path = f"{LOG_DIR}/{model_name}.tsv"
    with open(file_path, "w") as f:
        for task, metrics in results.items():
            for metric, values in metrics.items():
                f.write(f"{model_name}\t{task}\t{metric}\t{values}\n")
    print(f"Results saved to {file_path}")
    time_taken = time.time() - start
    hours, rem = divmod(time_taken, 3600)
    minutes, seconds = divmod(rem, 60)
    log_message = (
        f"Time taken for fine-tuning:\n"
        f"Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
        f"Total seconds: {time_taken:.2f}\n"
    )
    print(log_message)