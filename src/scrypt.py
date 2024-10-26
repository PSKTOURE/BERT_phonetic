import sys
import subprocess
import time
from config import DATASETS_DIR
from utils import convert_to_phonetic, download_bookcorpus
from dataset_cleaning import clean

# Download the bookcorpus dataset
start = time.time()
dataset = download_bookcorpus()
cleaned_dataset = clean(f"{DATASETS_DIR}/bookcorpus")
phonetic_dataset = convert_to_phonetic(f"{DATASETS_DIR}/cleaned_bookcorpus")
print(f"Finished in {time.time() - start:.2f} seconds")
subprocess.run(
    [
        sys.executable,
        "src/train_tokenizer.py",
        "--dataset_path",
        f"{DATASETS_DIR}/phonetic_cleaned_bookcorpus",
        "--tokenizer_type",
        "BPE",
        "--is_phonetic",
    ]
)
subprocess.run(
    [
        sys.executable,
        "src/bert_wikitext_train.py",
        "--dataset_path",
        f"{DATASETS_DIR}/phonetic_cleaned_bookcorpus",
        "--tokenizer_type",
        "BPE",
        "--fp16",
        "--num_epochs",
        "40",
        "--batch_size",
        "300",
        "--is_phonetic",
        "--custom_message",
        "Training on cleaned phonetic bookcorpus cleaned",
    ]
)
