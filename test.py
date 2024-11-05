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


model_path = "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1"
dataset_path = f"{DATASETS_DIR}/verses/phonetic_hf_rhymes"
tokenizer_path = "psktoure/BERT_WordPiece_phonetic_cleaned_wikitext-103-raw-v1"
task = "rhyme"
fine_tune_on_rhymes(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
    dataset_path=dataset_path,
    task_name=task,
)
#{'rhyme': {'accuracy': 0.7584950694051881}}
#{'rhyme': {'accuracy': 0.6679228691665097}}