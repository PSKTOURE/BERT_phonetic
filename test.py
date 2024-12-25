from src.train_on_all_task import fine_tune_on_all_tasks
from src.rhythm import predict_rhythm
from src.predict_last_verse_word import predict_word
import time



models = [
    "psktoure/BERT_TS_WordPiece_phonetic_wikitext_0.9"

]

is_phonetic = [True]

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/verses/phonetic_verse_hf"]

for path, dataset_path in zip(models, dataset_paths):
    predict_rhythm(
        model_path=path,
        dataset_path=dataset_path,
        batch_size=256,
        max_length=128,
        num_epochs=3,
        num_iterations=5,
        log_file="rhythm.tsv",
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/verses/rhyming_verses_phonetic"]
for path, dataset_path in zip(models, dataset_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=path,
        num_epochs=3,
        batch_size=256,
        num_iterations=5,
        max_length=128,
        k=5,
        log_file="predict_last_word.tsv",
    )

dataset_paths = ["/home/toure215/BERT_phonetic/DATASETS/rap/phonetic_rap_ds_hf"]
for path, dataset_path in zip(models, dataset_paths):
    predict_word(
        dataset_path=dataset_path,
        model_path=path,
        num_epochs=3,
        batch_size=256,
        num_iterations=5,
        max_length=128,
        k=5,
        log_file="predict_last_word.tsv",
    )
