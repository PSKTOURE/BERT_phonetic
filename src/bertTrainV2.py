from transformers import (
    PreTrainedTokenizerBase,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    BertConfig,
    AutoTokenizer,
    EarlyStoppingCallback,
)
import numpy as np
from datasets import load_from_disk
from typing import List, Dict, Tuple
import torch
import os
import re
import random
from src.config import MAX_LENGTH, MODEL_DIR, LOG_DIR, BATCH_SIZE, DEFAULT_MODEL
from src.utils import num_processes
from collections import defaultdict


class CustomDataCollatorForLanguageModeling:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int = 128,
        mask_probability: float = 0.15,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability
        self.cache = defaultdict(int)

    def _create_aligned_masks(
        self,
        normal_text: str,
        phonetic_text: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create masks following standard BERT masking strategy:
        - Select 15% of tokens for potential masking
        - Of those tokens:
            - 80% are replaced with [MASK]
            - 10% are replaced with random token
            - 10% are left unchanged
        Maintains alignment between normal and phonetic texts

        Args:
            normal_text: Original text
            phonetic_text: Phonetic transcription of the text

        Returns:
            Tuple containing:
            - normal_mask: Masking tensor for normal text
            - phonetic_mask: Masking tensor for phonetic text
            - normal_encoding: Token IDs for normal text
            - phonetic_encoding: Token IDs for phonetic text
        """
        # Split texts into words

        normal_words = re.findall(r"\w+|[^\w\s]", normal_text, re.UNICODE)
        phonetic_words = re.findall(r"\w+|[^\w\s]", phonetic_text, re.UNICODE)

        # Get token lengths for each word
        normal_token_lengths = [self._get_step_size(w) for w in normal_words]
        phonetic_token_lengths = [self._get_step_size(w) for w in phonetic_words]

        # Create cumulative sums for position mapping
        normal_cumsum = np.cumsum([0] + normal_token_lengths[:-1])
        phonetic_cumsum = np.cumsum([0] + phonetic_token_lengths[:-1])

        # Tokenize both texts
        normal_encoding = self.tokenizer(
            normal_text,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length // 2 - 2,
            return_tensors="pt",
        )["input_ids"]

        phonetic_encoding = self.tokenizer(
            phonetic_text,
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_length // 2 - 2,
            return_tensors="pt",
        )["input_ids"]

        # Initialize mask tensors (1 for MASK, 2 for random, 3 for unchanged)
        normal_mask = torch.zeros(normal_encoding.size(1), dtype=torch.long)
        phonetic_mask = torch.zeros(phonetic_encoding.size(1), dtype=torch.long)

        # Calculate number of words to mask (15% of the shorter sequence)
        num_words = min(len(normal_words), len(phonetic_words))
        num_to_mask = max(1, int(num_words * self.mask_probability))

        # Randomly select word positions to mask
        mask_indices = random.sample(range(num_words), num_to_mask)

        # Pre-calculate mask types for efficiency
        # 1: MASK, 2: random, 3: unchanged
        mask_types = np.random.choice([1, 2, 3], size=len(mask_indices), p=[0.8, 0.1, 0.1])

        # Apply masks
        for word_idx, mask_type in zip(mask_indices, mask_types):
            # Mask normal text
            normal_start = normal_cumsum[word_idx]
            normal_end = normal_start + normal_token_lengths[word_idx]
            normal_mask[normal_start:normal_end] = mask_type

            # Mask phonetic text
            phonetic_start = phonetic_cumsum[word_idx]
            phonetic_end = phonetic_start + phonetic_token_lengths[word_idx]
            phonetic_mask[phonetic_start:phonetic_end] = mask_type

        return normal_mask, phonetic_mask, normal_encoding, phonetic_encoding

    def _get_step_size(self, word: str) -> int:
        """return the number of tokens in a word"""
        if word in self.cache:
            return self.cache[word]
        tokens = self.tokenizer(word, add_special_tokens=False)["input_ids"]
        self.cache[word] = len(tokens)
        return self.cache[word]

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # Tokenize and process examples
        batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_labels = [], [], [], []

        for example in examples:
            normal_text = example["original_text"]
            phonetic_text = example["text"]

            # Create masks
            normal_mask, phonetic_mask, normal_encoding, phonetic_encoding = (
                self._create_aligned_masks(normal_text, phonetic_text)
            )

            # Combine normal and phonetic text
            final_input_ids = torch.cat(
                [
                    torch.tensor([self.tokenizer.cls_token_id]),  # [CLS]
                    normal_encoding[0],
                    torch.tensor([self.tokenizer.sep_token_id]),  # [SEP]
                    phonetic_encoding[0],
                    torch.tensor([self.tokenizer.sep_token_id]),  # Final [SEP]
                ],
            )

            # Create attention mask
            attention_mask = torch.ones(len(final_input_ids))

            # Create token type IDs
            # +1 for [SEP]
            normal_type_ids = torch.zeros(normal_encoding.size(1))
            phonetic_type_ids = torch.ones(phonetic_encoding.size(1))
            token_type_ids = torch.cat(
                [
                    torch.tensor([0]),
                    normal_type_ids,
                    torch.tensor([0]),
                    phonetic_type_ids,
                    torch.tensor([1]),
                ]
            )

            # Create labels
            labels = final_input_ids.clone()

            # Apply masks
            combined_mask = torch.cat(
                [
                    torch.tensor([0]),  # For [CLS]
                    normal_mask,
                    torch.tensor([0]),  # For [SEP]
                    phonetic_mask,
                    torch.tensor([0]),  # For final [SEP]
                ]
            )

            # Apply different masking strategies
            mask_token_id = self.tokenizer.mask_token_id
            for i in range(len(final_input_ids)):
                mask_type = combined_mask[i]
                if mask_type == 1:  # 80% - Replace with [MASK]
                    final_input_ids[i] = mask_token_id
                elif mask_type == 2:  # 10% - Replace with random token
                    final_input_ids[i] = np.random.randint(5, self.tokenizer.vocab_size)

            # Set labels
            labels = torch.where(combined_mask > 0, labels, -100)

            # Pad if necessary
            if len(final_input_ids) < self.max_length:
                padding_length = self.max_length - len(final_input_ids)
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length)])
                token_type_ids = torch.cat([token_type_ids, torch.zeros(padding_length)])
                labels = torch.cat([labels, torch.tensor([-100] * padding_length)])
                final_input_ids = torch.cat(
                    [
                        final_input_ids,
                        torch.tensor([self.tokenizer.pad_token_id] * padding_length),
                    ]
                )

            # Add to batch
            batch_input_ids.append(final_input_ids)
            batch_attention_masks.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(labels)

        # Stack tensors
        return {
            "input_ids": torch.stack(batch_input_ids).long(),
            "attention_mask": torch.stack(batch_attention_masks).long(),
            "token_type_ids": torch.stack(batch_token_type_ids).long(),
            "labels": torch.stack(batch_labels).long(),
        }


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


def train(
    dataset_path: str,
    tokenizer_path: str,
    num_epochs: int = 40,
    max_steps: int = -1,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-4,
    max_length: int = MAX_LENGTH,
    mask_probability: float = 0.15,
    fp16: bool = False,
    log_dir: str = LOG_DIR,
    model_dir: str = MODEL_DIR,
) -> Trainer:
    dataset_name = os.path.basename(dataset_path)
    print(f"Training BERT on dataset with tokenizer on {dataset_name} for {num_epochs} epochs")

    print("Preprocessing dataset ...")
    try:
        dataset = load_from_disk(dataset_path)
    except FileNotFoundError:
        raise ValueError(f"Dataset {dataset_path} not found")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        max_length=max_length,
        mask_probability=mask_probability,
    )

    config = setup_bert_config(vocab_size=tokenizer.vocab_size)
    model = BertForMaskedLM(config)
    model.resize_token_embeddings(len(tokenizer))

    hub_token = os.getenv("HF_TOKEN")
    model_name = f"BERT_IPA"
    training_args = TrainingArguments(
        output_dir=f"{model_dir}/{model_name}",
        overwrite_output_dir=True,
        remove_unused_columns=False,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
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
        resume_from_checkpoint="latest",
        dataloader_num_workers=num_processes,
        logging_dir=f"{log_dir}/tensorboard_{model_name}",
        logging_steps=100,
        report_to="tensorboard",
        seed=42,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=2_000,
        hub_token=hub_token,
        hub_model_id=model_name,
        push_to_hub=hub_token is not None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    print("Training ...")
    if os.listdir(f"{MODEL_DIR}/{model_name}"):
        print("Resuming from checkpoint ...")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    if hub_token is not None:
        trainer.push_to_hub("End of training")
    return trainer
