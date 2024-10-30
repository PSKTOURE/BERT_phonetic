from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_from_disk
from src.train_tokenizer import load_tokenizer
import os

dataset = load_from_disk("DATASETS/wikitext-103-raw-v1-test")
tokenizer = load_tokenizer("tokenizers/tokenizer_BPE")
data_collator_mlm = DataCollatorWithPadding(tokenizer=tokenizer)
hub_token = os.getenv("HF_TOKEN")
print(hub_token)
dataset_tokenized = dataset.map(
    lambda x: tokenizer(x["text"], padding="max_length", truncation=True),
    batched=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    "/home/toure215/BERT_phonetic/models/BERT_BPE_wikitext-103-raw-v1-test/checkpoint-1364"
)
training_args = TrainingArguments(
    output_dir=f"/home/toure215/BERT_phonetic/models/BERT_BPE_wikitext-103-raw-v1-test/checkpoint-1364",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=256,
    per_device_eval_batch_size=256,
    learning_rate=1e-5,
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
    dataloader_num_workers=16,
    seed=42,
    fp16=True,
    eval_strategy="steps",
    eval_steps=2_000,
    gradient_accumulation_steps=4,
    hub_token=hub_token,
    hub_model_id="BERT_BPE_wikitext-103-raw-v1-test",
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
trainer.push_to_hub("End of training")
