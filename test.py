from src.train_on_all_task import fine_tune_on_all_tasks

models = [
    "psktoure/BERT_TS_WordPiece_phonetic_wikitext" 
]
is_phonetic = [True]

for path, p in zip(models, is_phonetic):
    fine_tune_on_all_tasks(
        num_iterations=5,
        model_path=path,
        is_phonetic=p,
        all=True,
    )
