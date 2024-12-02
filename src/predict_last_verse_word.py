import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    Trainer,
    TrainingArguments,
)
import epitran
from transformers import PreTrainedTokenizerBase
from collections import defaultdict
from src.config import LOG_DIR


epi = epitran.Epitran("eng-Latn")
num_processes = os.cpu_count() - 1


class CustomDataCollator:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, padding=True, max_length=128
    ):
        self.tokenizer = tokenizer
        self.mask_token_id = tokenizer.mask_token_id
        self.padding = padding
        self.max_length = max_length

    def __call__(self, examples):

        sentence1 = [example["sentence1"] for example in examples]
        sentence2 = [example["sentence2"] for example in examples]
        targets = [str(example["label"]) for example in examples]

        encoded_targets = self.tokenizer(
            targets,
            add_special_tokens=False,
        )["input_ids"]

        batch = self.tokenizer(
            sentence1,
            sentence2,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        for i, idx in enumerate(input_ids):
            sep_token_indices = torch.where(idx == self.tokenizer.sep_token_id)[0]
            start = sep_token_indices[0] - len(encoded_targets[i])
            end = sep_token_indices[0]
            input_ids[i, start:end] = self.mask_token_id
            labels[i, :start] = -100
            labels[i, end:] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }
    


def predict(
    dataset_path: str,
    model_path: str,
    num_epochs: int = 3,
    batch_size: int = 256,
    num_iterations: int = 5,
    k: int = 5,
    log_file: str = "rap_predict_last_word.tsv",
):
    
    def evaluate_rhyme(model, dataset, tokenizer):
        model = model.to("cuda")
        model.eval()
        res = []

        for i in range(0, len(dataset), batch_size):
            print(f"Processing batch {i}/{len(dataset)}...", end="\r")
            batch = dataset[i : i + batch_size]
            batch_sequence = [{key: batch[key][j] for key in batch} for j in range(len(batch["sentence1"]))]
            inputs = data_collator(batch_sequence)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                labels = inputs["labels"]

            count = 0

            for j in range(len(batch["sentence1"])):
                masked_token_index = (inputs["input_ids"][j] == tokenizer.mask_token_id).nonzero(as_tuple=True)[0]
                targets = labels[j, masked_token_index]
                top_k_indices = logits[j, masked_token_index].topk(k).indices.squeeze(0)
                if i < 16 and j < 8:
                    print('targets:', targets, '-- top_k_indices:', top_k_indices)

                # Check if the target index is in the top-k predictions
                ok = True
                for idx, target in enumerate(targets):
                    if target not in top_k_indices[idx]:
                        ok = False
                if ok:
                    count += 1

            res.append(count / len(batch["sentence1"]))

        return np.mean(res)

    def one_iteration_training(num_iter: int):
        model = AutoModelForMaskedLM.from_pretrained(model_path)
        
        training_args = TrainingArguments(
            output_dir="/tmp/fine_tuned_bert",
            eval_strategy="epoch",
            save_strategy="no",
            learning_rate=5e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=num_epochs,
            weight_decay=0.01,
            logging_strategy='no',
            remove_unused_columns=False,
            seed=np.random.randint(1e6),
            fp16=True,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            data_collator=data_collator,
        )

        print(f"Training {model_name} on {num_iter+1}/{num_iterations} iterations")
        trainer.train()

        return evaluate_rhyme(model, dataset['test'], tokenizer)
    
    try:
        dataset = load_from_disk(dataset_path)
    except:
        raise ValueError(f"Dataset {dataset_path} not found")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data_collator = CustomDataCollator(tokenizer)
    
    model_name = model_path.split("/")[-1]
    
    results = defaultdict(list)

    for i in range(num_iterations):
        accuracy = one_iteration_training(i)
        results["accuracy"].append(accuracy)
    
    results["accuracy"] = {
        "mean": np.mean(results["accuracy"]),
        "std": np.std(results["accuracy"]),
    }

    print(f"Results for {model_name}: {results}")
    with open(f"{LOG_DIR}/{log_file}", "a") as f:
        f.write(f"{model_name}\taccuracy\t{results['accuracy']}\n")


