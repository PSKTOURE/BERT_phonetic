from src.utils import translate_to_phonetic
from datasets import load_from_disk

example = {"text": "This is a test."}
print(translate_to_phonetic(example))
dataset = load_from_disk("DATASETS/phonetic_wikitext-103-raw-v1")
print(dataset)
print(dataset["train"][:10])
dataset = dataset.filter(lambda x: len(x["text"]) > 0)
print(dataset)
print(dataset["train"][:10])