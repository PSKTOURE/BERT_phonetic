from src.predict_last_verse_word import predict
from src.config import DATASETS_DIR

model_paths = [
    # "bert-base-uncased",
    # "psktoure/BERT_WordPiece_wikitext-103-raw-v1",
    # "psktoure/BERT_WordPiece_phonetic_wikitext-103-raw-v1",
    # "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1",
    # "psktoure/BERT_BPE_phonetic_wikitext-103-raw-v1",
    # "psktoure/BERT_BPE_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_WordLevel_phonetic_wikitext-103-raw-v1"
]


# dataset_paths = [f"{DATASETS_DIR}/verses/rhyming_verses"] * 2 + [f"{DATASETS_DIR}/verses/rhyming_verses_phonetic"] * 5
# is_phonetic = [False] * 2 + [True] * 5
dataset_paths = [f"{DATASETS_DIR}/verses/rhyming_verses_phonetic"]
is_phonetic = [True]


for model_path, dataset_path, is_phonetic in zip(model_paths, dataset_paths, is_phonetic):
    predict(
        model_path=model_path, 
        dataset_path=dataset_path, 
        num_iterations=5,
        batch_size=256,
        num_epochs=3,
        is_phonetic=is_phonetic
    )

