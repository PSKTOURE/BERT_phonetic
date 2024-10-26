from transformers import Trainer, TrainingArguments
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)
from src.train_on_all_task import (
    preprocess_dataset,
    compute_metrics,
)
from src.utils import task_to_num_labels, task_to_metric, load_glue_dataset_from_dir
import numpy as np
import optuna
import os

num_processes = os.cpu_count()


def search(
    model_path: str, 
    task: str, 
    n_trials: int = 5, 
    n_jobs: int = 1, 
    is_phonetic=False
):
    # Load the dataset
    dataset = load_glue_dataset_from_dir(task, is_phonetic=is_phonetic)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Tokenize the dataset
    tokenized_dataset = preprocess_dataset(dataset, tokenizer, task)

    datacollator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    num_labels = task_to_num_labels[task]

    # Define objective function for hyperparameter search
    def objective(trial):
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels, ignore_mismatched_sizes=True
        )

        training_args = TrainingArguments(
            output_dir="./results",
            save_strategy="no",
            evaluation_strategy="no",
            logging_strategy="no",
            per_device_train_batch_size=trial.suggest_categorical(
                "per_device_train_batch_size", [16, 32]
            ),
            per_device_eval_batch_size=8,
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True),
            num_train_epochs=trial.suggest_int("num_train_epochs", 2, 4),
            fp16=True,
            gradient_accumulation_steps=4,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=datacollator,
        )

        # Train the model
        trainer.train()
        res = compute_metrics(trainer, tokenized_dataset, task)
        metric = task_to_metric[task][0]
        if task == "mnli":
            matched_val = res["validation_matched"][metric]
            mismatched_val = res["validation_mismatched"][metric]
            return np.mean([matched_val, mismatched_val])
        return res[task][metric]

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Perform hyperparameter search
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Retrieve the best trial
    best_trial = study.best_trial

    # Print the best trial
    print("Best trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    return best_trial

if __name__ == "__main__":
    search("psktoure/BERT_WordPiece_wikitext-103-raw-v1", "rte", 4, 1)