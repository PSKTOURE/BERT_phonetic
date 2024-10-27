# Define import
from itertools import chain
import os
import argparse
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
)

from transformers import BertConfig, BertForMaskedLM
from transformers import DataCollatorForLanguageModeling
from src.config import MAX_LENGTH, MODEL_DIR, LOG_DIR, BATCH_SIZE
from src.utils import num_processes, timeit
from src.train_tokenizer import load_tokenizer

########### BERT TRAINING START HERE ############
# First tokenized wikitest dataset
def preprocess_dataset(dataset, tokenizer, max_length=MAX_LENGTH) -> dict:
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=max_length,
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=num_processes,
    )

    return tokenized_dataset


# Define a BERT configuration
def setup_bert_config(
    vocab_size: int,
    hidden_size: int = 768,
    num_hidden_layers: int = 12,
    num_attention_heads: int = 12,
    intermediate_size: int = 3072,
    max_position_embeddings: int = 512,
) -> BertConfig:
    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size,
        max_position_embeddings=max_position_embeddings,
    )
    return config


# BERT model
def train(
    dataset_path: str,
    tokenizer_type: str = "WordPiece",
    num_epochs: int = 40,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-4,
    max_length: int = MAX_LENGTH,
    fp16: bool = False,
    is_phonetic: bool = False,
    log_dir: str = LOG_DIR,
    model_dir: str = MODEL_DIR,
) -> Trainer:
    dataset_name = os.path.basename(dataset_path)
    print(
        f"Training BERT with {tokenizer_type} tokenizer on dataset {dataset_name} for {num_epochs} epochs"
    )
    tokenizer = load_tokenizer(tokenizer_type, is_phonetic)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    print("Preprocessing dataset ...")
    try:
        dataset = load_from_disk(dataset_path)
    except:
        raise ValueError(f"Dataset {dataset_path} not found")
    dataset_tokenized = preprocess_dataset(dataset, tokenizer, max_length=max_length)

    config = setup_bert_config(vocab_size=vocab_size)
    model = BertForMaskedLM(config)

    data_collator_mlm = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )
    hub_token = os.getenv("HF_TOKEN")
    model_name = f"BERT_{tokenizer_type}_{dataset_name}"
    training_args = TrainingArguments(
        output_dir=f"{model_dir}/{model_name}",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-6,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        save_steps=2_000,
        warmup_steps=10_000,
        save_total_limit=1,
        load_best_model_at_end=True,
        resume_from_checkpoint=True,
        dataloader_num_workers=num_processes,
        logging_dir=f"{log_dir}/tensorboard_{model_name}",
        logging_steps=100,
        report_to="tensorboard",
        seed=42,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=2_000,
        gradient_accumulation_steps=4,
        hub_token=hub_token,
        hub_model_id=f"{model_name}",
        push_to_hub=hub_token is not None,
        # gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator_mlm,
        train_dataset=dataset_tokenized["train"],
        eval_dataset=dataset_tokenized["validation"],
    )
    print("Training ...")
    if os.listdir(f"{MODEL_DIR}/{model_name}"):
        print("Resuming from checkpoint ...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.push_to_hub("End of training")
    tokenizer.push_to_hub(model_name)
    return trainer

