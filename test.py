from src.homophones import fine_tune_on_homophones
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
is_phonetic = [False, False, True, True, True, True, True]

dataset_path = f"{DATASETS_DIR}/homophones_data/hf_dataset"

for model_path, phonetic in zip(model_paths, is_phonetic):
    fine_tune_on_homophones(
        model_path=model_path, 
        dataset_path=dataset_path, 
        tokenizer_path=model_path, 
        num_iterations=5, 
        batch_size=256,
        num_epochs=3,
        is_phonetic=phonetic
    )

