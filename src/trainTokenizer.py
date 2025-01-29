import os
import shutil
from datasets import load_from_disk, concatenate_datasets
from tokenizers import Tokenizer
from tokenizers.models import WordLevel, BPE, WordPiece
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer, WordPieceTrainer, BpeTrainer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from transformers import PreTrainedTokenizerFast
from src.config import DATASETS_DIR, TOKENIZERS_DIR, BERT_DEFAULT_VOCAB_SIZE


def get_training_corpus(dataset, is_phonetic: bool):
    splits = list(dataset.keys())
    dataset = concatenate_datasets([dataset[split] for split in splits])
    text = "text" if is_phonetic else "original_text"
    for i in range(0, len(dataset), 1000):
        samples = dataset[i : i + 1000]
        yield samples["text"] + [" "] + samples["original_text"]


def train_tokenizer(
    dataset_path, tokenizer_type: str, is_phonetic: bool
) -> PreTrainedTokenizerFast:
    try:
        dataset = load_from_disk(dataset_path)
        print(dataset)
    except:
        raise ValueError(f"Dataset {dataset_path} not found")

    if not os.path.exists(TOKENIZERS_DIR):
        os.makedirs(TOKENIZERS_DIR)

    prefix = "phonetic_" if is_phonetic else ""
    tokenizer_name = f"tokenizer_{prefix}{tokenizer_type}_IPA"
    if os.path.exists(f"{TOKENIZERS_DIR}/{tokenizer_name}"):
        print("Removing existing tokenizer directory ...")
        shutil.rmtree(f"{TOKENIZERS_DIR}/{tokenizer_name}")

    tokenizer_dict = {"WordLevel": WordLevel, "BPE": BPE, "WordPiece": WordPiece}
    tokenizer_trainers = {
        "WordLevel": WordLevelTrainer,
        "WordPiece": WordPieceTrainer,
        "BPE": BpeTrainer,
    }

    if tokenizer_type not in tokenizer_dict:
        raise ValueError(f"Invalid tokenizer type: {tokenizer_type}")
    tokenizer = Tokenizer(tokenizer_dict[tokenizer_type](unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = tokenizer_trainers[tokenizer_type](
        vocab_size=BERT_DEFAULT_VOCAB_SIZE,
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
    )
    training_corpus = get_training_corpus(dataset, is_phonetic)
    tokenizer.train_from_iterator(training_corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
    tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "mask_token": "[MASK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
        }
    )
    tokenizer.save_pretrained(f"{TOKENIZERS_DIR}/{tokenizer_name}")
    print(f"Tokenizer saved to {TOKENIZERS_DIR}/{tokenizer_name}")
    return tokenizer


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer_name = os.path.basename(tokenizer_path)
    print(f"Loading tokenizer from {tokenizer_name}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer
