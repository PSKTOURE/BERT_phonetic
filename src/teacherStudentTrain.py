import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    BertForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    BertConfig,
    AutoModel,
    EarlyStoppingCallback,
)
from torch.utils.tensorboard import SummaryWriter
from src.config import BATCH_SIZE, MAX_LENGTH, LOG_DIR, MODEL_DIR, DEFAULT_MODEL
from src.utils import num_processes


def teacher_student_training(
    dataset_path: str,
    tokenizer_path: str,
    teacher_model_name: str = DEFAULT_MODEL,
    num_epochs: int = 40,
    max_steps: int = -1,
    batch_size: int = BATCH_SIZE,
    lr: float = 1e-4,
    max_length: int = MAX_LENGTH,
    d_lambda: float = 0.1,
    fp16: bool = False,
    tokenizer_type: str = "WordPiece",
    log_dir: str = LOG_DIR,
    model_dir: str = MODEL_DIR,
    inverse: bool = False,
):

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
            output_hidden_states=True,
        )
        return config
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models and move to device
    teacher = AutoModel.from_pretrained(teacher_model_name)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_model_name)
    teacher.eval()

    for param in teacher.parameters():
        param.requires_grad = False
    
    # If inverse is False, the student model is the phonetic model; otherwise, it is the BERT base model
    if not inverse:
        # Student is the phonetic model
        student_tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        config = setup_bert_config(vocab_size=student_tokenizer.vocab_size)
        student = BertForMaskedLM(config=config)
    else:
        # Student is the bert-base model
        student = BertForMaskedLM.from_pretrained(DEFAULT_MODEL, output_hidden_states=True)
        student_tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    
    teacher.to(device)
    student.to(device)

    class TeacherStudentTrainer(Trainer):
        def __init__(self, teacher=None, student=None, d_lambda: float = 0.1, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.teacher = teacher
            self.student = student
            self.d_lambda = d_lambda
            self.mse = torch.nn.MSELoss()
            self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir)

        def compute_loss(self, model, inputs, return_outputs=False):
            teacher_input_ids = inputs.pop("teacher_input_ids")
            teacher_attention_mask = inputs.pop("teacher_attention_mask")
            student_input_ids = inputs.pop("student_input_ids")
            student_attention_mask = inputs.pop("student_attention_mask")

            with torch.no_grad():
                teacher_outputs = self.teacher(
                    input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
                )
                teacher_cls_embeddings = teacher_outputs.last_hidden_state[:, 0, :]  # [CLS]

            student_outputs = model(
                input_ids=student_input_ids,
                attention_mask=student_attention_mask,
                labels=inputs["labels"],
            )
            student_cls_embeddings = student_outputs.hidden_states[-1][:, 0, :]  # [CLS]

            # Losses
            mlm_loss = student_outputs.loss
            mse_loss = self.mse(student_cls_embeddings, teacher_cls_embeddings)

            mse_loss_value = mse_loss.item()
            mlm_loss_value = mlm_loss.item()

            scale_factor = mlm_loss_value / mse_loss_value if mse_loss_value != 0 else 0
            total_loss = (1 - self.d_lambda) * mlm_loss + self.d_lambda * scale_factor * mse_loss

            # Log losses to TensorBoard
            current_step = self.state.global_step
            if current_step % 500 == 0:
                self.tb_writer.add_scalar("Loss/MLM", mlm_loss.item(), current_step)
                self.tb_writer.add_scalar("Loss/MSE", mse_loss.item(), current_step)

            return (total_loss, student_outputs) if return_outputs else total_loss
        
        def on_train_end(self, *args, **kwargs):
            super().on_train_end(*args, **kwargs)
            self.tb_writer.close() 

    mlm_data_collator = DataCollatorForLanguageModeling(
        tokenizer=student_tokenizer, mlm=True, mlm_probability=0.15
    )

    def tokenize_dataset(dataset, teacher_tokenizer, student_tokenizer, max_length=max_length):
        teacher_text = "original_text"
        student_text = "text"
        if inverse:
            teacher_text, student_text = student_text, teacher_text

        def tokenize_function(examples):
            teacher_inputs = teacher_tokenizer(
                examples[teacher_text],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            student_inputs = student_tokenizer(
                examples[student_text],
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            return {
                "teacher_input_ids": teacher_inputs["input_ids"],
                "teacher_attention_mask": teacher_inputs["attention_mask"],
                "student_input_ids": student_inputs["input_ids"],
                "student_attention_mask": student_inputs["attention_mask"],
            }

        return dataset.map(tokenize_function, batched=True, num_proc=num_processes)
    
    def data_collator(batch):
        # Extract pre-tokenized data
        teacher_input_ids = torch.tensor([example["teacher_input_ids"] for example in batch])
        teacher_attention_mask = torch.tensor([example["teacher_attention_mask"] for example in batch])
        student_input_ids = torch.tensor([example["student_input_ids"] for example in batch])
        student_attention_mask = torch.tensor([example["student_attention_mask"] for example in batch])
        
        # Apply MLM masking to student inputs
        student_inputs = mlm_data_collator(
            [{"input_ids": input_id} for input_id in student_input_ids]
        )
        
        return {
            "teacher_input_ids": teacher_input_ids,
            "teacher_attention_mask": teacher_attention_mask,
            "student_input_ids": student_inputs["input_ids"],
            "student_attention_mask": student_attention_mask,
            "labels": student_inputs["labels"],  # MLM labels
        }

    # Load dataset
    dataset = load_from_disk(dataset_path)
    tokenized_dataset = tokenize_dataset(dataset, teacher_tokenizer, student_tokenizer, max_length=max_length)
    dataset_name = os.path.basename(dataset_path)

    hub_token = os.getenv("HF_TOKEN")
    if not inverse:
        model_name = f"BERT_TS_{tokenizer_type}_{dataset_name}_{d_lambda}"
    else:
        model_name = f"BERT_BASE_TS_{dataset_name}_{d_lambda}"

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
        logging_steps=500,
        report_to="tensorboard",
        seed=42,
        fp16=fp16,
        eval_strategy="steps",
        eval_steps=2_000,
        hub_token=hub_token,
        hub_model_id=model_name,
        push_to_hub=hub_token is not None,
    )

    trainer = TeacherStudentTrainer(
        teacher=teacher,
        student=student,
        d_lambda=d_lambda,
        model_init=lambda: student,
        args=training_args,
        data_collator=data_collator,
        tokenizer=student_tokenizer,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    print("Training ...")
    if os.listdir(f"{MODEL_DIR}/{model_name}"):
        print("Resuming from checkpoint ...")
        trainer.train(resume_from_checkpoint=True)
    else:
        if inverse:
            print("################# TRAINING BERT BASE AS STUDENT #################")
        trainer.train()
    if hub_token is not None:
        trainer.push_to_hub("End of training")
    return trainer
