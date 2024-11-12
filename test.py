from custom_task import fine_tune_on_task
from src.config import DATASETS_DIR

model_paths = [
    "bert-base-uncased",
    "psktoure/BERT_WordPiece_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_WordLevel_phonetic_wikitext-103-raw-v1"
]


dataset_paths = [f"{DATASETS_DIR}/etymology/etymology_pairs_hf"] * 2 + [f"{DATASETS_DIR}/etymology/etymology_pairs_hf_phonetic"] * 5


for model_path, dataset_path in zip(model_paths, dataset_paths):
    fine_tune_on_task(
        model_path=model_path, 
        dataset_path=dataset_path, 
        tokenizer_path=model_path, 
        num_iterations=5, 
        batch_size=512,
        num_epochs=3,
        use_roc=False,
        log_file="etymology_results.tsv"
    )

