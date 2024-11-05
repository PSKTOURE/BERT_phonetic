from src.train_on_all_task import fine_tune_on_rhymes
from src.config import DATASETS_DIR
from datasets import load_from_disk
from src.utils import translate_task_to_phonetic

# dataset_path = f"{DATASETS_DIR}/verses/hf_rhymes"
# dataset = load_from_disk(dataset_path)
# dataset = dataset.map(lambda x: translate_task_to_phonetic(x, "rhyme"), num_proc=15, batched=True)
# dataset.save_to_disk(f"{DATASETS_DIR}/verses/phonetic_hf_rhymes", num_proc=4)
# print(dataset)
# print(dataset["train"][0])


model_paths = [
    "psktoure/BERT_BPE_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_BPE_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1",
    "psktoure/BERT_WordPiece_phonetic_wikitext-103-raw-v1",
    "psktoure/BERT_WordLevel_phonetic_wikitext-103-raw-v1",
]
dataset_path = f"{DATASETS_DIR}/verses/phonetic_hf_rhymes"
task = "rhyme"
for model_path in model_paths:
    print(f"Fine-tuning {model_path} on {task}")
    fine_tune_on_rhymes(
        model_path=model_path,
        dataset_path=dataset_path,
        tokenizer_path=model_path,
        task_name=task
    )
#bert-base : {'rhyme': {'accuracy': 0.7584950694051881}}
#bert_WP_pho_cleaned : {'rhyme': {'accuracy': 0.6679228691665097}}
#bert_WP_ph : {'rhyme': {'accuracy': 0.6338797814207651}}
#bert_BPE_pho: {'rhyme': {'accuracy': 0.6793543119150807}}
#bert_BPE_pho_cleaned : {'rhyme': {'accuracy': 0.6524087682934488}}
