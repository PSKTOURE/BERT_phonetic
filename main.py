import os
import time
import shutil
import argparse
from collections import defaultdict
from src.config import DATASETS_DIR, TOKENIZERS_DIR, DEFAULT_MODEL
from src.utils import convert_to_phonetic, download_bookcorpus, timeit, task_to_num_labels
from src.bert_wikitext_train import train
from src.dataset_cleaning import clean
from src.train_on_all_task import fine_tune_on_all_tasks
from src.train_tokenizer import train_tokenizer

# Initialize argparse for main commands
parser = argparse.ArgumentParser()
parser.add_argument("train", action="store_true", help="Launch training process")
parser.add_argument(
    "fine_tune", action="store_true", help="Launch fine-tuning process"
)
parser.add_argument(
    "train_tokenizer", action="store_true", help="Launch tokenizer training"
)
args = parser.parse_args()

# Load default config arguments from config.txt
default_args = {
    # Training default arguments
    "tm::dataset_path": f"{DATASETS_DIR}/phonetic_cleaned_bookcorpus",
    "tm::tokenizer_type": "BPE",
    "tm::tokenizer_path": f"{TOKENIZERS_DIR}/tokenizer_phonetic_BPE",
    "tm::num_epochs": "40",
    "tm::fp16": "TRUE",
    "tm::batch_size": "256",
    "tm::lr": "0.0001",
    "tm::max_length": "128",
    "tm::log_dir": "logs",
    "tm::model_dir": "models",
    # Fine-tuning default arguments
    "ft::model_path": DEFAULT_MODEL,
    "ft::all": "FALSE",
    "ft::n": "1",
    "ft::is_phonetic": "TRUE",
    # Tokenizer training default arguments
    "tt::dataset_path": f"{DATASETS_DIR}/phonetic_cleaned_bookcorpus",
    "tt::tokenizer_type": "BPE",
    "tt::is_phonetic": "TRUE",
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
config_args["ft::is_phonetic"] = config_args["ft::is_phonetic"].upper() == "TRUE"
config_args["tt::is_phonetic"] = config_args["tt::is_phonetic"].upper() == "TRUE"
config_args["tm::fp16"] = config_args["tm::fp16"].upper() == "TRUE"
config_args["ft::all"] = config_args["ft::all"].upper() == "TRUE"

# Execute the appropriate command
if args.train:
    start = time.time()
    if "phonetic_bookcorpus" not in os.listdir(DATASETS_DIR):
        print("Downloading bookcorpus dataset and converting to phonetic ...")
        download_bookcorpus(is_phonetic=True)

    # Train the model using config arguments
    timeit(train)(
        dataset_path=config_args["tm::dataset_path"],
        tokenizer_type=config_args["tm::tokenizer_type"],
        tokenizer_path=config_args["tm::tokenizer_path"],
        num_epochs=int(config_args["tm::num_epochs"]),
        fp16=config_args["tm::fp16"],
        batch_size=int(config_args["tm::batch_size"]),
        lr=float(config_args["tm::lr"]),
        max_length=int(config_args["tm::max_length"]),
        log_dir=config_args["tm::log_dir"],
        model_dir=config_args["tm::model_dir"],
    )

elif args.fine_tune:
    fine_tune_on_all_tasks(
        model_path=config_args["ft::model_path"],
        is_phonetic=config_args["ft::is_phonetic"],
        task_to_num_labels=task_to_num_labels,
        all=config_args["ft::all"],
        n=int(config_args["ft::n"]),
        tokenizer_path=config_args["ft::tokenizer_path"],
    )

elif args.train_tokenizer:
    train_tokenizer(
        config_args["tt::dataset_path"],
        config_args["tt::tokenizer_type"],
        is_phonetic=config_args["tt::is_phonetic"],
    )
