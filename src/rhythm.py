import os
import torch
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_from_disk
from src.config import LOG_DIR
from torchmetrics.classification import MulticlassAccuracy
from sklearn.utils.class_weight import compute_class_weight
from collections import defaultdict


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_train = self.train_dataset["label"]

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = torch.tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(self.y_train),
                y=self.y_train,
            ),
            device=self.args.device,
            dtype=torch.float,
        )
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


def fine_tune(
    model_path: str,
    dataset_path: str,
    batch_size: int = 256,
    max_length: int = 256,
    num_epochs: int = 3,
    num_iterations: int = 5,
    log_file: str = "rhythm.tsv",
):

    def tokenize_function(examples):
        return tokenizer(
            examples["Verse"], padding=False, truncation=True, max_length=max_length
        )

    dataset = load_from_disk(dataset_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_tokenized = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=os.cpu_count() - 1,
    )
    data_collator = DataCollatorWithPadding(tokenizer)
    model_name = model_path.split("/")[-1]

    def _fine_tune(num_iter: int):
        print(f"Fine-tuning {model_name} for {num_iter + 1}/{num_iterations} iteration")

        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=4, ignore_mismatched_sizes=True)
        training_args = TrainingArguments(
            output_dir="/tmp/ryhthm_bert",
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_strategy="epoch",
            logging_strategy="no",
            save_strategy="no",
            report_to="none",
            seed=np.random.randint(1e6),
            fp16=True,
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset_tokenized["train"],
            eval_dataset=dataset_tokenized["validation"],
            data_collator=data_collator,
        )

        trainer.train()

        predictions = trainer.predict(dataset_tokenized["test"])
        preds, labels = predictions.predictions, predictions.label_ids
        preds = np.argmax(preds, axis=-1)
        accuracy = MulticlassAccuracy(num_classes=4, average="macro")
        acc = accuracy(torch.tensor(preds), torch.tensor(labels))
        return acc

    res = defaultdict(list)
    for i in range(num_iterations):
        acc = _fine_tune(i)
        res["accuracy"].append(acc)
    
    res["accuracy"] = {
        "mean": np.mean(res["accuracy"]),
        "std": np.std(res["accuracy"]),
    }

    with open(f"{LOG_DIR}/{log_file}", "a") as f:
        f.write(f"{model_name}\taccuracy\t{res['accuracy']}\n")