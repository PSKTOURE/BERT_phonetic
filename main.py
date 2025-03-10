import argparse
from collections import defaultdict
from src.config import DATASETS_DIR, TOKENIZERS_DIR, DEFAULT_MODEL
from src.utils import (
    download_bookcorpus,
    download_wikitext,
    download_glue_dataset,
    timeit,
)
from src.bertTrainV1 import train
from src.bertTrainV2 import train as trainV2
from src.teacherStudentTrain import teacher_student_training
from src.fineTuneOnGlue import fine_tune_on_all_tasks
from src.trainTokenizer import train_tokenizer

# Initialize argparse for main commands
parser = argparse.ArgumentParser()
parser.add_argument(
    "--download_bookcorpus",
    action="store_true",
    help="Download bookcorpus dataset",
)
parser.add_argument(
    "--download_wikitext",
    action="store_true",
    help="Download wikitext dataset",
)
parser.add_argument(
    "--download_glue",
    action="store_true",
    help="Download GLUE datasets",
)
parser.add_argument(
    "--is_phonetic",
    action="store_true",
    help="Convert downloaded datasets to phonetic",
)
parser.add_argument("--train", action="store_true", help="Launch training process")
parser.add_argument("--trainV2", action="store_true", help="Launch training process")

parser.add_argument(
    "--distillation_training", action="store_true", help="Launch distillation training process"
)
parser.add_argument("--fine_tune", action="store_true", help="Launch fine-tuning process")
parser.add_argument("--train_tokenizer", action="store_true", help="Launch tokenizer training")
args = parser.parse_args()

# Load default config arguments from config.txt
default_args = {
    # Training default arguments
    "tm::dataset_path": f"{DATASETS_DIR}/phonetic_wikitext",
    "tm::tokenizer_type": "BPE",
    "tm::tokenizer_path": f"{TOKENIZERS_DIR}/XSAMPA/tokenizer_phonetic_WordPiece",
    "tm::teacher_model_name": DEFAULT_MODEL,
    "tm::num_epochs": "40",
    "tm::max_steps": "-1",
    "tm::fp16": "TRUE",
    "tm::batch_size": "256",
    "tm::lr": "0.0001",
    "tm::max_length": "128",
    "tm::d_lambda": "0.1",
    "tm::inverse": "FALSE",
    "tm::log_dir": "logs",
    "tm::model_dir": "models",
    "tm::percent": "0.1",
    # Fine-tuning default arguments
    "ft::model_path": DEFAULT_MODEL,
    "ft::all": "FALSE",
    "ft::num_iterations": "1",
    "ft::is_phonetic": "TRUE",
    # Tokenizer training default arguments
    "tt::dataset_path": f"{DATASETS_DIR}/phonetic_wikitext",
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
            key, value = line.split("==", 1)
            config_args[key.strip()] = value.strip()
except FileNotFoundError:
    print("Config file not found, using default arguments.")

# Convert string arguments to appropriate types
config_args["ft::is_phonetic"] = config_args["ft::is_phonetic"].upper() == "TRUE"
config_args["tt::is_phonetic"] = config_args["tt::is_phonetic"].upper() == "TRUE"
config_args["tm::fp16"] = config_args["tm::fp16"].upper() == "TRUE"
config_args["tm::inverse"] = config_args["tm::inverse"].upper() == "TRUE"
config_args["ft::all"] = config_args["ft::all"].upper() == "TRUE"

print("Config arguments:")
for key, value in config_args.items():
    print(f"{key}: {value}")

# Check if the datasets are downloaded
if args.download_bookcorpus:
    download_bookcorpus(is_phonetic=args.is_phonetic)

elif args.download_wikitext:
    download_wikitext(is_phonetic=args.is_phonetic)

elif args.download_glue:
    download_glue_dataset(is_phonetic=args.is_phonetic)

# Execute the appropriate command
elif args.train:
    # Train the model using config arguments
    timeit(train)(
        dataset_path=config_args["tm::dataset_path"],
        tokenizer_type=config_args["tm::tokenizer_type"],
        tokenizer_path=config_args["tm::tokenizer_path"],
        num_epochs=int(config_args["tm::num_epochs"]),
        max_steps=int(config_args["tm::max_steps"]),
        fp16=config_args["tm::fp16"],
        batch_size=int(config_args["tm::batch_size"]),
        lr=float(config_args["tm::lr"]),
        max_length=int(config_args["tm::max_length"]),
        log_dir=config_args["tm::log_dir"],
        model_dir=config_args["tm::model_dir"],
    )

elif args.trainV2:
    # Train the model on both phonetic and normal datasets
    timeit(trainV2)(
        dataset_path=config_args["tm::dataset_path"],
        tokenizer_path=config_args["tm::tokenizer_path"],
        num_epochs=int(config_args["tm::num_epochs"]),
        max_steps=int(config_args["tm::max_steps"]),
        batch_size=int(config_args["tm::batch_size"]),
        lr=float(config_args["tm::lr"]),
        max_length=int(config_args["tm::max_length"]),
        fp16=config_args["tm::fp16"],
        log_dir=config_args["tm::log_dir"],
        model_dir=config_args["tm::model_dir"],
    )

elif args.distillation_training:
        # Launch distillation training process
        timeit(teacher_student_training)(
            dataset_path=config_args["tm::dataset_path"],
            tokenizer_path=config_args["tm::tokenizer_path"],
            teacher_model_name=config_args["tm::teacher_model_name"],
            num_epochs=int(config_args["tm::num_epochs"]),
            max_steps=int(config_args["tm::max_steps"]),
            batch_size=int(config_args["tm::batch_size"]),
            lr=float(config_args["tm::lr"]),
            max_length=int(config_args["tm::max_length"]),
            d_lambda=float(config_args["tm::d_lambda"]),
            inverse=config_args["tm::inverse"],
            fp16=config_args["tm::fp16"],
            tokenizer_type=config_args["tm::tokenizer_type"],
            log_dir=config_args["tm::log_dir"],
            model_dir=config_args["tm::model_dir"],
        )

elif args.fine_tune:
    fine_tune_on_all_tasks(
        model_path=config_args["ft::model_path"],
        is_phonetic=config_args["ft::is_phonetic"],
        all=config_args["ft::all"],
        num_iterations=int(config_args["ft::num_iterations"]),
    )

elif args.train_tokenizer:
    train_tokenizer(
        config_args["tt::dataset_path"],
        config_args["tt::tokenizer_type"],
        is_phonetic=config_args["tt::is_phonetic"],
    )
