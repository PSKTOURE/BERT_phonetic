import os

# Configuration file for the project
MAX_LENGTH = 128
BATCH_SIZE = 32
HOME = os.environ["HOME"]
DATASETS_DIR = "DATASETS"
MODEL_DIR = "models"
LOG_DIR = "logs"
ORIGINAL_DIR = "original"
PHONETIC_DIR = "phonetic"
GLUE_DIR = "glue_data"
DEFAULT_MODEL = "bert-base-uncased"
BERT_DEFAULT_VOCAB_SIZE = 30522
TOKENIZERS_DIR = "tokenizers"
GLUE_TASKS = ["cola", "mnli", "mnli_matched", "mnli_mismatched", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
REPO_ID = "psktoure"