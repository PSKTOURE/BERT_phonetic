import os
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import evaluate
import epitran
from collections import defaultdict 
from scipy.special import softmax
import numpy as np
from functools import lru_cache
import multiprocessing
from src.config import DATASETS_DIR, LOG_DIR

num_proc = multiprocessing.cpu_count() - 1
dataset_path = f"{DATASETS_DIR}/homophones_data/hf_dataset"
epi = epitran.Epitran("eng-Latn")

def fine_tune_on_homophones(
        model_path: str, 
        dataset_path: str, 
        tokenizer_path: str, 
        num_iterations: int =1, 
        batch_size:int = 256,
        num_epochs:int = 3,
        is_phonetic=False
    ):
    @lru_cache(maxsize=None)
    def xsampa(word):
        return "".join(epi.xsampa_list(word))
        
    def translate_to_phonetic(example):
        return {
            "word1": xsampa(example["word1"]),
            "word2": xsampa(example["word2"]),
            "label": example["label"]
        }

    dataset = load_from_disk(dataset_path)
    if is_phonetic:
        dataset = dataset.map(translate_to_phonetic, num_proc=num_proc)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    

    def tokenize_function(examples):
        return tokenizer(
            examples["word1"], 
            examples["word2"], 
            padding=False, 
            truncation=True, 
            max_length=128
        )
    
    encoded_dataset = dataset.map(tokenize_function, batched=True)
    data_collector = DataCollatorWithPadding(tokenizer=tokenizer)
    roc = evaluate.load("roc_auc")
    model_name = model_path.split("/")[-1]

    def _fine_tune_on_homophones(iteration):
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

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
            seed=np.random.randint(1e6)
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["validation"],
            data_collator=data_collector
        )

        print(f"##################Iteration {iteration+1}/{num_iterations}##################")
        trainer.train()
        predictions = trainer.predict(encoded_dataset["test"])
        preds, labels = predictions.predictions, predictions.label_ids
        pred_scores = softmax(preds, axis=1)[:, 1]
        roc_score = roc.compute(references=labels, prediction_scores=pred_scores)
        return roc_score["roc_auc"]

    results = defaultdict(list)
    for i in range(num_iterations):
        roc_score = _fine_tune_on_homophones(i)
        results["roc_auc"].append(roc_score)
    
    results["roc_auc"] = {
        "mean": np.mean(results["roc_auc"]),
        "std": np.std(results["roc_auc"]),
    }
  
    print(f"Results for {model_name}: {results}")
    with open(f"{LOG_DIR}/homophones_results.tsv", "a") as f:
        f.write(f"{model_name}\troc_auc\t{results['roc_auc']}\n")






