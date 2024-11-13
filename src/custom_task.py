import os
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
import evaluate
import epitran
from collections import defaultdict
from scipy.special import softmax
import numpy as np
import multiprocessing
from src.config import DATASETS_DIR, LOG_DIR

task_to_fields = {
    "homophones": ("word1", "word2"),
    "rhyme": ("sentence1", "sentence2"),
    "etymology": ("word1", "word2"),
    "verses": ("Verse",)
}
task_to_num_labels = {
    "homophones": 2,
    "rhyme": 2,
    "etymology": 2,
    "verses": 4
}

num_proc = multiprocessing.cpu_count() - 1
dataset_path = f"{DATASETS_DIR}/homophones_data/hf_dataset"
epi = epitran.Epitran("eng-Latn")


def fine_tune_on_task(
    model_path: str,
    dataset_path: str,
    tokenizer_path: str,
    num_iterations: int = 1,
    batch_size: int = 256,
    num_epochs: int = 3,
    use_roc: bool = False,
    task: str = "homophones",
    log_file: str = "homophones_results.tsv",
):
    dataset = load_from_disk(dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def tokenize_function(examples):
        fields = task_to_fields[task]
        if len(fields) == 1:
            return tokenizer(examples[fields[0]], padding=False, truncation=True, max_length=128)
        return tokenizer(
            examples[fields[0]],
            examples[fields[1]],
            padding=False,
            truncation=True,
            max_length=128,
        )
        
    encoded_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=task_to_fields[task] ,
    )
    data_collector = DataCollatorWithPadding(tokenizer=tokenizer)
    
    model_name = model_path.split("/")[-1]
    metric = "roc_auc" if use_roc else "accuracy"
    num_labels = task_to_num_labels[task]

    def _fine_tune_on_task(iteration):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir=f"/tmp/{model_name}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            evaluation_strategy="epoch",
            logging_strategy="no",
            save_strategy="no",
            overwrite_output_dir=True,
            fp16=True,
            seed=np.random.randint(1e6),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            data_collator=data_collector,
        )

        print(
            f"##################{model_name} on Iteration {iteration+1}/{num_iterations}##################"
        )
        trainer.train()
        predictions = trainer.predict(encoded_dataset["test"])
        preds, labels = predictions.predictions, predictions.label_ids
        if use_roc:
            roc = evaluate.load("roc_auc")
            pred_scores = softmax(preds, axis=1)[:, 1]
            roc_score = roc.compute(references=labels, prediction_scores=pred_scores)
            score = roc_score[metric]
        else:
            preds = np.argmax(preds, axis=-1)
            score = np.mean(preds == labels)
        return score

    results = defaultdict(list)
    for i in range(num_iterations):
        score = _fine_tune_on_task(i)
        results[metric].append(score)

    results[metric] = {
        "mean": np.mean(results[metric]),
        "std": np.std(results[metric]),
    }

    print(f"Results for {model_name}: {results}")
    with open(f"{LOG_DIR}/{log_file}", "a") as f:
        f.write(f"{model_name}\t{metric}\t{results[metric]}\n")
