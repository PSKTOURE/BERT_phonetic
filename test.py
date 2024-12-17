from src.train_on_all_task import fine_tune_on_all_tasks
import time

time.sleep(60*60*3)

models = [
    "psktoure/BERT_BPE_wikitext",
    "psktoure/BERT_BPE_phonetic_wikitext" 
]
is_phonetic = [False, True]

for path, p in zip(models, is_phonetic):
    fine_tune_on_all_tasks(
        num_iterations=5,
        model_path=path,
        is_phonetic=p,
        all=True,
    )
