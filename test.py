from src.fineTuneOnGlue import fine_tune_on_all_tasks
from src.rhythm import predict_rhythm
from src.predictLastWord import predict_word
import time


models = [
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.01",
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.3",
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.5",
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.7",
    "psktoure/BERT_BASE_TS_phonetic_wikitext_0.9",
    "psktoure/BERT_IPA",
]


for model in models:
    fine_tune_on_all_tasks(
        model_path=model,
        num_iterations=3,
        is_phonetic=False,
        all=True,
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/verses/verse_hf"]

for path, dataset_path in zip(models, dataset_paths):
    predict_rhythm(
        model_path=path,
        dataset_path=dataset_path,
        batch_size=256,
        max_length=128,
        num_epochs=3,
        num_iterations=3,
        log_file="rhythm.tsv",
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/verses/rhyming_verses"]
for path, dataset_path in zip(models, dataset_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=path,
        num_epochs=3,
        batch_size=256,
        num_iterations=3,
        max_length=128,
        k=5,
        log_file="predict_last_word.tsv",
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/rap/rap_ds_hf"]
for path, dataset_path in zip(models, dataset_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=path,
        num_epochs=3,
        batch_size=256,
        num_iterations=3,
        max_length=128,
        k=5,
        log_file="rap_predict_last_word.tsv",
    )
