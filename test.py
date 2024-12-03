from src.predict_last_verse_word import predict
from src.config import DATASETS_DIR

model_paths = [
    "bert-base-uncased",
    "psktoure/BERT_WordPiece_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_cleaned_wikitext-103-raw-v1",
]


dataset_paths = [f"{DATASETS_DIR}/rap/rap_ds_hf"] * 2 + [f"{DATASETS_DIR}/rap/phonetic_rap_ds_hf"] * 4


for model_path, dataset_path in zip(model_paths, dataset_paths):
    try:
        predict(
            model_path=model_path,
            dataset_path=dataset_path,
            num_iterations=3,
            batch_size=32,
            num_epochs=3,
        )
    except Exception as e:
        print(f"Error with model {model_path} and dataset {dataset_path}: {e}")
