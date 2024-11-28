from src.rhythm import fine_tune
from src.config import DATASETS_DIR

# model_paths = [
#     "bert-base-uncased",
#     "psktoure/BERT_WordPiece_wikitext-103-raw-v1",
#     "psktoure/BERT_WordPiece_phonetic_wikitext-103-raw-v1",
#     "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1",
#     "psktoure/BERT_BPE_phonetic_wikitext-103-raw-v1",
#     "psktoure/BERT_BPE_phonetic_cleaned_wikitext-103-raw-v1",
# ]


# dataset_paths = [f"{DATASETS_DIR}/verses/verses_hf"] * 2 + [f"{DATASETS_DIR}/verses/phonetic_verses_hf"] * 4


# for model_path, dataset_path in zip(model_paths, dataset_paths):
#     fine_tune(
#         model_path=model_path, 
#         dataset_path=dataset_path, 
#         num_iterations=5,
#         batch_size=256,
#         num_epochs=3,
#         max_length=256,
#     )
    
from datasets import load_from_disk

dataset = load_from_disk(f"{DATASETS_DIR}/phonetic_bookcorpus")
print(dataset)
print(dataset["train"][:10])
