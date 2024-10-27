import re
import os
import hashlib
import time
import multiprocessing
import argparse
from datasets import load_from_disk
from lingua import Language, LanguageDetectorBuilder

num_processes = multiprocessing.cpu_count()
# Exact duplication removal (on individual sentences/paragraphs)
def remove_exact_duplicates(examples):
    seen = set()
    deduped_examples = []
    for sentence in examples["text"]:
        hash_val = hashlib.md5(sentence.encode()).hexdigest()
        if hash_val not in seen:
            seen.add(hash_val)
            deduped_examples.append(sentence)
    return {"text": deduped_examples}


def filter_by_language(examples):
    detector = LanguageDetectorBuilder.from_languages(Language.ENGLISH, Language.FRENCH).build()
    return {
        "text": [
            sentence for sentence in examples["text"] if detector.detect_language_of(sentence) == Language.ENGLISH
        ]
    }


# Basic text cleaning
def clean_text(examples):
    cleaned_text = []
    for sentence in examples["text"]:
        # Lowercase
        sentence = sentence.lower()
        # Remove extra spaces
        sentence = re.sub(r"\s+", " ", sentence)
        # Remove URLs
        sentence = re.sub(r"http\S+", "", sentence)
        # Remove special characters
        sentence = re.sub(r"[^a-zA-Z0-9,.!?;:\'\" ]+", "", sentence)
        cleaned_text.append(sentence.strip())
    return {"text": cleaned_text}

def clean(dataset_path: str):
    start = time.time()
    try:
        dataset = load_from_disk(dataset_path)
    except:
        raise ValueError(f"Dataset {dataset_path} not found")
    dataset = dataset.map(remove_exact_duplicates, batched=True, num_proc=num_processes)
    dataset = dataset.map(filter_by_language, batched=True, num_proc=num_processes)
    dataset = dataset.map(clean_text, batched=True, num_proc=num_processes)
    dataset_dir = os.path.dirname(dataset_path)
    cleaned_dataset_path = f"{dataset_dir}/cleaned_{os.path.basename(dataset_path)}"
    dataset.save_to_disk(cleaned_dataset_path, num_proc=num_processes)
    print(f"Cleaning took {time.time() - start:.2f} seconds")
    print(f"Cleaned dataset saved to {cleaned_dataset_path}")
    return cleaned_dataset_path

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--dataset_path", type=str, required=True)
    args = args.parse_args()
    clean(args.dataset_path)

