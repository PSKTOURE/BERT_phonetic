import os
import time
import shutil
import argparse
from collections import defaultdict
from src.config import DATASETS_DIR, DEFAULT_MODEL
from src.utils import convert_to_phonetic, download_bookcorpus, task_to_num_labels
from src.bert_wikitext_train import train
from src.dataset_cleaning import clean
from src.train_on_all_task import fine_tune_on_all_tasks
from src.train_tokenizer import train_tokenizer

# Initialize argparse for main commands
parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true", help="Launch training process")
parser.add_argument(
    "--fine_tune", action="store_true", help="Launch fine-tuning process"
)
parser.add_argument(
    "--train_tokenizer", action="store_true", help="Launch tokenizer training"
)
args = parser.parse_args()

# Load default config arguments from config.txt
default_args = {
    "--dataset_path": f"{DATASETS_DIR}/phonetic_cleaned_bookcorpus",
    "--tokenizer_type": "BPE",
    "--is_phonetic": "TRUE",
    "--num_epochs": "40",
    "--fp16": "TRUE",
    "--batch_size": "256",
    "--lr": "0.0001",
    "--max_length": "128",
    "--log_dir": "logs",
    "--model_dir": "models",
    "--custom_message": "Training BERT phonetic on bookcorpus",
    "--model_path": DEFAULT_MODEL,
    "--all": "FALSE",
    "--n": "1",
    "--tokenizer_path": None,
}
config_args = defaultdict(lambda: None, default_args)

# Read config file values and update defaults
try:
    with open("config.txt", "r") as f:
        for line in f:
            if line.startswith("#") or line == "\n":
                continue
            key, value = line.split(" ", 1)
            config_args[key.strip()] = value.strip()
except FileNotFoundError:
    print("Config file not found, using default arguments.")

# Convert string arguments to appropriate types
config_args["--is_phonetic"] = config_args["--is_phonetic"].upper() == "TRUE"
config_args["--fp16"] = config_args["--fp16"].upper() == "TRUE"
config_args["--all"] = config_args["--all"].upper() == "TRUE"
config_args["--tokenizer_path"] = (
    None
    if config_args["--tokenizer_path"].lower() == "None".lower()
    else config_args["--tokenizer_path"]
)

# Execute the appropriate command
if args.train:
    start = time.time()
    if "phonetic_cleaned_bookcorpus" not in os.listdir(DATASETS_DIR):
        download_bookcorpus()
        clean(f"{DATASETS_DIR}/bookcorpus")
        convert_to_phonetic(f"{DATASETS_DIR}/cleaned_bookcorpus")
        shutil.rmtree(f"{DATASETS_DIR}/bookcorpus")
        shutil.rmtree(f"{DATASETS_DIR}/cleaned_bookcorpus")
    print(f"Finished preprocessing in {time.time() - start:.2f} seconds")

    # Train the model using config arguments
    train(
        dataset_path=config_args["--dataset_path"],
        tokenizer_type=config_args["--tokenizer_type"],
        num_epochs=int(config_args["--num_epochs"]),
        batch_size=int(config_args["--batch_size"]),
        lr=float(config_args["--lr"]),
        max_length=int(config_args["--max_length"]),
        fp16=config_args["--fp16"],
        is_phonetic=config_args["--is_phonetic"],
        log_dir=config_args["--log_dir"],
        model_dir=config_args["--model_dir"],
    )

elif args.fine_tune:
    fine_tune_on_all_tasks(
        model_path=config_args["--model_path"],
        is_phonetic=config_args["--is_phonetic"],
        task_to_num_labels=task_to_num_labels,
        all=config_args["--all"],
        n=int(config_args["--n"]),
        tokenizer_path=config_args["--tokenizer_path"],
    )

elif args.train_tokenizer:
    train_tokenizer(
        config_args["--dataset_path"],
        config_args["--tokenizer_type"],
        is_phonetic=config_args["--is_phonetic"],
    )
