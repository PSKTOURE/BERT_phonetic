import os
import re
import time
import multiprocess
import epitran
from functools import lru_cache
from datasets import DatasetDict, load_dataset, concatenate_datasets, load_from_disk
from src.datasetCleaning import (
    remove_exact_duplicates,
    filter_by_language,
    clean_text,
)
from src.config import (
    DATASETS_DIR,
    GLUE_DIR,
    ORIGINAL_DIR,
    PHONETIC_DIR,
    GLUE_TASKS,
)

epi = epitran.Epitran("eng-Latn")
num_processes = multiprocess.cpu_count() - 1

task_to_fields = {
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence",),
    "qqp": ("question1", "question2"),
    "mnli": ("premise", "hypothesis"),
    "mnli_matched": ("premise", "hypothesis"),
    "mnli_mismatched": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "cola": ("sentence",),
    "mrpc": ("sentence1", "sentence2"),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_num_labels = {
    "rte": 2,
    "sst2": 2,
    "qqp": 2,
    "mnli": 3,
    "qnli": 2,
    "cola": 2,
    "mrpc": 2,
    "stsb": 1,
    "wnli": 2,
}
task_to_metric = {
    "rte": ["accuracy"],
    "sst2": ["accuracy"],
    "qqp": ["f1"],
    "mnli": ["accuracy"],
    "qnli": ["accuracy"],
    "cola": ["accuracy"],
    "mrpc": ["f1"],
    "stsb": ["spearmanr"],
    "wnli": ["accuracy"],
}

task_to_path = {task: f"{DATASETS_DIR}/{GLUE_DIR}/{ORIGINAL_DIR}/{task}" for task in GLUE_TASKS}
task_to_path_phonetic = {
    task: f"{DATASETS_DIR}/{GLUE_DIR}/{PHONETIC_DIR}/{task}" for task in GLUE_TASKS
}


def download_glue_dataset(is_phonetic=False):
    for task in GLUE_TASKS:
        print(f"Downloading {task} dataset")
        dataset = load_dataset("glue", task)
        if is_phonetic:
            path = task_to_path_phonetic[task]
            dataset = dataset.map(
                lambda x: translate_task_to_phonetic(x, task),
                num_proc=num_processes,
                batched=True,
            )
        else:
            path = task_to_path[task]
        # save dataset
        dataset.save_to_disk(path, num_proc=num_processes)


def load_glue_dataset_from_dir(task, is_phonetic=False):
    if is_phonetic:
        path = task_to_path_phonetic[task]
    else:
        path = task_to_path[task]
    if not os.path.exists(path):
        return load_dataset("glue", task)
    # load from disk
    if task != "mnli":
        return load_dataset(path, num_proc=num_processes)
    mnli = load_dataset(path, num_proc=num_processes)
    mnli_matched = load_dataset(f"{path}_matched", num_proc=num_processes)
    mnli_mismatched = load_dataset(f"{path}_mismatched", num_proc=num_processes)
    mnli["validation_matched"] = mnli_matched["validation"]
    mnli["validation_mismatched"] = mnli_mismatched["validation"]
    mnli["test_matched"] = mnli_matched["test"]
    mnli["test_mismatched"] = mnli_mismatched["test"]
    del mnli["validation"]
    del mnli["test"]
    return mnli


def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)

        elapsed_time = time.time() - start
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        tokenizer_type = kwargs.get("tokenizer_type", "WordPiece")
        dataset_path = kwargs.get("dataset_path", "None")
        dataset_name = os.path.basename(dataset_path)

        log_dir = kwargs.get("log_dir", ".")
        os.makedirs(log_dir, exist_ok=True)

        log_message = (
            f"Training operation for BERT_{tokenizer_type}_{dataset_name}:\n"
            f"Elapsed time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n"
            f"Total seconds: {elapsed_time:.2f}\n"
        )

        print(log_message)

        # Write to log file
        log_file_path = os.path.join(log_dir, "training_time.txt")
        with open(log_file_path, "a") as log_file:
            log_file.write(log_message + "\n")

        return result

    return wrapper


# Download wikitext-v3
def download_wikitext(is_phonetic=False) -> None:
    start = time.time()
    dataset_name = "wikitext-103-raw-v1"
    print(f"Downloading wikitext dataset: {dataset_name} ...")

    if dataset_name not in os.listdir(DATASETS_DIR):
        print("Downloading dataset...")
        dataset = load_dataset("Salesforce/wikitext", dataset_name, num_proc=num_processes)
    else:
        dataset = load_from_disk(f"{DATASETS_DIR}/{dataset_name}")

    dataset = (
        dataset.map(lambda x: chunked_text(x, chunk_size=50), num_proc=num_processes, batched=True)
        .map(clean_text, num_proc=num_processes, batched=True)
        .map(remove_exact_duplicates, num_proc=num_processes, batched=True)
        .map(filter_by_language, num_proc=num_processes, batched=True)
        .filter(lambda x: len(x["text"]) > 0, num_proc=num_processes)
        .flatten_indices(num_proc=num_processes)
    )
    dataset_name = f"wikitext"

    if is_phonetic:
        dataset = dataset.map(translate_to_phonetic, num_proc=num_processes, batched=True).filter(
            lambda x: len(x["text"]) > 0, num_proc=num_processes
        )
        print("Translating to phonetic...")

        dataset_name = f"phonetic_wikitext2"

    wiki_dir = f"{DATASETS_DIR}/{dataset_name}"
    dataset.save_to_disk(wiki_dir, num_proc=num_processes)
    print(dataset)
    elapsed_time = time.time() - start
    print(f"Downloaded {dataset_name} dataset in {elapsed_time:.2f} seconds")
    print(f"Saved dataset to {wiki_dir}")


def chunked_text(examples, chunk_size=100):
    all = []
    for sentence in examples["text"]:
        words = re.findall(r"\w+|[^\s\w]+", sentence)
        all += [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return {"text": all}


def download_bookcorpus(is_phonetic=False) -> None:
    start = time.time()
    if "bookcorpus" not in os.listdir(DATASETS_DIR):
        bookcorpus = load_dataset(
            "bookcorpus",
            trust_remote_code=True,
            split="train",
            num_proc=num_processes,
        )
        wiki = load_dataset(
            "wikipedia",
            "20220301.en",
            split="train",
            trust_remote_code=True,
            num_proc=num_processes,
        )
        wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
        assert bookcorpus.features.type == wiki.features.type
        bookcorpus = concatenate_datasets([bookcorpus, wiki])
        bookcorpus = bookcorpus.train_test_split(test_size=1e-2)
        bookcorpus = DatasetDict({"train": bookcorpus["train"], "validation": bookcorpus["test"]})

        bookcorpus = (
            bookcorpus.map(
                lambda x: chunked_text(x, chunk_size=100),
                batched=True,
                num_proc=num_processes,
            )
            .flatten_indices(num_proc=num_processes)
            .filter(lambda x: len(x["text"]) > 0, num_proc=num_processes)
            .map(clean_text, batched=True, num_proc=num_processes)
            .shuffle(seed=42)
        )
    else:
        bookcorpus = load_from_disk(f"{DATASETS_DIR}/bookcorpus")

    if is_phonetic:
        bookcorpus = bookcorpus.map(translate_to_phonetic, batched=True, num_proc=num_processes)
    prefix = "phonetic_" if is_phonetic else ""
    bookcorpus_folder = f"{DATASETS_DIR}/{prefix}bookcorpus/"
    bookcorpus.save_to_disk(bookcorpus_folder, num_proc=num_processes)
    print(f"Saved bookcorpus dataset to {bookcorpus_folder} in {time.time() - start:.2f} seconds")


def convert_to_phonetic(dataset_path):
    start = time.time()
    dataset = load_from_disk(dataset_path)
    dataset = dataset.map(translate_to_phonetic, num_proc=num_processes, batched=True)
    data_dir = os.path.dirname(dataset_path)
    phonetic_dataset_path = f"{data_dir}/phonetic_{os.path.basename(dataset_path)}"
    dataset.save_to_disk(phonetic_dataset_path, num_proc=num_processes)
    print(f"Converted dataset to phonetic in {time.time() - start:.2f} seconds")
    return dataset


@lru_cache(maxsize=None)
def cached_xsampa(word):
    return "".join(epi.xsampa_list(word))


def xsampa(sentences):
    # return [
    #     " ".join(cached_xsampa(word) for word in re.findall(r"\w+|[^\s\w]+", sentence))
    #     for sentence in sentences
    # ]
    return [epi.transliterate(sentence) for sentence in sentences]

def translate_to_phonetic(example):
    phonetic_sentences = xsampa(example["text"])
    return {"text": phonetic_sentences, "original_text": example["text"]}


def translate_task_to_phonetic(example, task_name):
    fields = task_to_fields.get(task_name, None)

    if not fields:
        raise ValueError(f"Task {task_name} not found in task_to_fields dictionary.")

    if len(fields) == 1:
        # sst2 case
        example[fields[0]] = xsampa(example[fields[0]])
    else:
        # the rest hopefully
        example[fields[0]] = xsampa(example[fields[0]])
        example[fields[1]] = xsampa(example[fields[1]])

    return example


if __name__ == "__main__":
    pass
