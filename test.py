from src.fineTuneOnGlue import fine_tune_on_all_tasks
from src.rhythm import predict_rhythm
from src.predictLastWord import predict_word
import time


models = [
    "bert-base-uncased",
    "psktoure/BERT_WordPiece_wikitext",
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.1",
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.5",
    "psktoure/BERT_WordPiece_phonetic_wikitext",
    "psktoure/BERT_TS_WordPiece_phonetic_wikitext_0.1",
    "psktoure/BERT_TS_WordPiece_phonetic_wikitext_0.5",
    
]


# for model in models:
#     fine_tune_on_all_tasks(
#         model_path=model,
#         num_iterations=3,
#         is_phonetic=True,
#         all=True,
#     )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/verses/verses_hf"] * 4 + ["/home/toure215/BERT_phonetic/DATASETS/verses/phonetic_verses_hf"] * 3

for path, dataset_path in zip(models, dataset_paths):
    predict_rhythm(
        model_path=path,
        dataset_path=dataset_path,
        batch_size=64,
        max_length=128,
        num_epochs=3,
        num_iterations=5,
        log_file="rhythm.tsv",
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/verses/rhyming_verses"] * 4 + ["/home/toure215/BERT_phonetic/DATASETS/verses/rhyming_verses_phonetic"] * 3
for path, dataset_path in zip(models, dataset_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=path,
        num_epochs=3,
        batch_size=16,
        num_iterations=5,
        max_length=128,
        k=5,
        log_file="predict_last_word.tsv",
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/rap/rap_ds_rhyme"] * 4 + ["/home/toure215/BERT_phonetic/DATASETS/rap/phonetic_rap_ds_hf"] * 3
for path, dataset_path in zip(models, dataset_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=path,
        num_epochs=3,
        batch_size=16,
        num_iterations=5,
        max_length=128,
        k=5,
        log_file="rap_predict_last_word.tsv",
    )
