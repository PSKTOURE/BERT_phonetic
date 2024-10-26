import re
import os
import hashlib
import time
import argparse
from datasets import load_from_disk
from utils import num_processes
from langdetect import detect, LangDetectException
from lingua import Language, LanguageDetectorBuilder


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


# Language filtering
# def filter_by_language(examples, lang="en"):
#     return {
#         "text": [
#             sentence for sentence in examples["text"] if detect_language(sentence) == lang
#         ]
#     }


# def detect_language(text):
#     try:
#         return detect(text)
#     except LangDetectException:
#         return "unknown"

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
    # dataset_path = f"/home/toure215/DATASETS/cleaned_wikitext-103-raw-v1"
    # original_dataset_path = f"/home/toure215/DATASETS/wikitext-103-raw-v1"
    # original_dataset = load_from_disk(original_dataset_path)
    # dataset = load_from_disk(dataset_path)
    # print(dataset)
    # print(original_dataset["validation"]["text"][:10])
    # print(dataset["validation"]["text"][:10])  

