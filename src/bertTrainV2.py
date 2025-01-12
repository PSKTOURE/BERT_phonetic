from transformers import (
    PreTrainedTokenizerBase,
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
    Trainer,
    TrainingArguments,
    BertConfig,
    AutoTokenizer,
)
from datasets import load_from_disk
from typing import List, Dict, Tuple
import torch
import os
import random
from src.config import MAX_LENGTH, MODEL_DIR, LOG_DIR, BATCH_SIZE, DEFAULT_MODEL
from src.utils import num_processes


class CustomDataCollatorForLanguageModeling():
    def __init__(
        self,
        normal_tokenizer: PreTrainedTokenizerBase,
        phonetic_tokenizer: PreTrainedTokenizerBase,
        max_length: int = MAX_LENGTH,
        mask_probability: float = 0.15,
    ):
        self.normal_tokenizer = normal_tokenizer
        self.phonetic_tokenizer = phonetic_tokenizer
        self.max_length = max_length
        self.mask_probability = mask_probability

    def _create_aligned_masks(
        self,
        normal_tokens: List[str],
        phonetic_tokens: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize masks
        normal_mask = torch.zeros(len(normal_tokens), dtype=torch.bool)
        phonetic_mask = torch.zeros(len(phonetic_tokens), dtype=torch.bool)

        # Track word positions (non-subword tokens)
        normal_word_positions = [
            i for i, token in enumerate(normal_tokens) if not token.startswith("##")
        ]
        phonetic_word_positions = [
            i for i, token in enumerate(phonetic_tokens) if not token.startswith("##")
        ]

        # Create word-level masks
        for i in range(min(len(normal_word_positions), len(phonetic_word_positions))):
            if random.random() < self.mask_probability:
                # Mask word in normal text
                normal_pos = normal_word_positions[i]
                normal_mask[normal_pos] = True
                j = normal_pos + 1
                while j < len(normal_tokens) and normal_tokens[j].startswith("##"):
                    normal_mask[j] = True
                    j += 1

                # Mask corresponding word in phonetic text
                phonetic_pos = phonetic_word_positions[i]
                phonetic_mask[phonetic_pos] = True
                j = phonetic_pos + 1
                while j < len(phonetic_tokens) and phonetic_tokens[j].startswith("##"):
                    phonetic_mask[j] = True
                    j += 1

        return normal_mask, phonetic_mask

    def __call__(self, examples: List[Dict[str, str]]) -> Dict[str, torch.Tensor]:
        # Tokenize and process examples
        batch_input_ids, batch_attention_masks, batch_token_type_ids, batch_labels = [], [], [], []

        for example in examples:
            normal_text = example["original_text"]
            phonetic_text = example["text"]

            # Tokenize both texts
            normal_encoding = self.normal_tokenizer(
                normal_text, 
                truncation=True,
                max_length=self.max_length // 2, 
                return_tensors="pt"
            )
            phonetic_encoding = self.phonetic_tokenizer(
                phonetic_text, 
                truncation=True,
                max_length=self.max_length // 2, 
                return_tensors="pt"
            )

            # Convert tokens for masking
            normal_tokens = self.normal_tokenizer.convert_ids_to_tokens(
                normal_encoding["input_ids"][0], skip_special_tokens=True
            )
            phonetic_tokens = self.phonetic_tokenizer.convert_ids_to_tokens(
                phonetic_encoding["input_ids"][0], skip_special_tokens=True
            )

            # Create aligned masks
            normal_mask, phonetic_mask = self._create_aligned_masks(normal_tokens, phonetic_tokens)

            # Combine normal and phonetic text
            final_input_ids = torch.cat(
                [
                    torch.tensor([self.normal_tokenizer.cls_token_id]),  # [CLS]
                    normal_encoding["input_ids"][0][1:-1],  # without [CLS] and [SEP]
                    torch.tensor([self.normal_tokenizer.sep_token_id]),  # [SEP]
                    phonetic_encoding["input_ids"][0][1:-1],  # without [CLS] and [SEP]
                    torch.tensor([self.normal_tokenizer.sep_token_id]),  # Final [SEP]
                ],
            )

            # Create attention mask
            attention_mask = torch.ones(len(final_input_ids))

            # Create token type IDs
            # +1 for [SEP]
            normal_type_ids = torch.zeros(len(normal_encoding["input_ids"][0][1:-1]) + 1)
            # +1 for [SEP]
            phonetic_type_ids = torch.ones(len(phonetic_encoding["input_ids"][0][1:-1]) + 1)
            token_type_ids = torch.cat([torch.tensor([0]), normal_type_ids, phonetic_type_ids])

            # Create labels (copy of input_ids)
            labels = final_input_ids.clone()

            # Apply masks
            combined_mask = torch.cat(
                [
                    torch.tensor([False]),  # For [CLS]
                    normal_mask,
                    torch.tensor([False]),  # For [SEP]
                    phonetic_mask,
                    torch.tensor([False]),  # For final [SEP]
                ]
            )
    
            final_input_ids[combined_mask] = self.normal_tokenizer.mask_token_id

            # Pad if necessary
            if len(final_input_ids) < self.max_length:
                padding_length = self.max_length - len(final_input_ids)
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length)])
                token_type_ids = torch.cat([token_type_ids, torch.zeros(padding_length)])
                labels = torch.cat([labels, torch.tensor([-100] * padding_length)])
                final_input_ids = torch.cat([final_input_ids, torch.tensor([self.normal_tokenizer.pad_token_id] * padding_length)])

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
    phonetic_tokenizer_path: str,
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
    except:
        raise ValueError(f"Dataset {dataset_path} not found")

    normal_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    phonetic_tokenizer = AutoTokenizer.from_pretrained(phonetic_tokenizer_path)

    data_collator = CustomDataCollatorForLanguageModeling(
        normal_tokenizer=normal_tokenizer,
        phonetic_tokenizer=phonetic_tokenizer,
        max_length=max_length,
        mask_probability=mask_probability,
    )

    config = setup_bert_config(vocab_size=normal_tokenizer.vocab_size)
    model = BertForMaskedLM(config)
    model.resize_token_embeddings(len(normal_tokenizer))

    hub_token = os.getenv("HF_TOKEN")
    model_name = f"BERT_V2"
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
        tokenizer=normal_tokenizer,
        data_collator=data_collator,
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
