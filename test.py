from src.predict_last_verse_word import predict_word
from src.rhythm import predict_rhythm
from src.utils import DATASETS_DIR

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/rap/rap_ds_rhyme"] * 2 + ["/home/toure215/BERT_phonetic/DATASETS/rap/phonetic_rap_ds_hf"] * 5
model_paths = [
    "bert-base-uncased",
    "psktoure/BERT_WordPiece_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_bookcorpus_0.1"
]

for dataset_path, model_path in zip(dataset_paths, model_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=model_path,
        num_epochs=3,
        batch_size=256,
        num_iterations=5,
        k=5,
        log_file='rap_predict_last_word.tsv'
    )

dataset_paths = [f"{DATASETS_DIR}/verses/verses_hf"] * 2 + [f"{DATASETS_DIR}/verses/phonetic_verses_hf"] * 5

for dataset_path, model_path in zip(dataset_paths, model_paths):
    predict_rhythm(
        dataset_path=dataset_path,
        model_path=model_path,
        num_epochs=3,
        batch_size=256,
        num_iterations=5,
    )