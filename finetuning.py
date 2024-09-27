import os
import torch
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig, 
    Trainer, 
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from trl import SFTTrainer


# Configuration
def create_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


# Load the model and tokenizer
def load_model_and_tokenizer(model_id, cache_dir, config=None, bnb_config=None):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=bnb_config,
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",  # optimized for memory
        cache_dir=cache_dir
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    return model, tokenizer


# Pad token setup
def setup_pad_token(tokenizer, model):
    if '<pad>' not in tokenizer.get_vocab():
        tokenizer.add_tokens(['<pad>'])
        tokenizer.pad_token = '<pad>'
        model.pad_token_id = tokenizer.pad_token_id
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)


# Make parameters trainable
def set_trainable_params(model, param_names):
    for name, param in model.named_parameters():
        if any(k in name for k in param_names):
            param.requires_grad_(True)


# Print parameter summary
def print_trainable_parameters_summary(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params}")
    print(f"Total parameters: {total_params}")
    print(f"Trainable percentage: {100 * trainable_params / total_params:.2f}%")


# Custom callback to log metrics
class LoggingCallback(transformers.TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        with open(self.log_file_path, 'a') as f:
            if 'loss' in logs:
                f.write(f"Step: {state.global_step}, Training Loss: {logs['loss']}\n")
            if 'eval_loss' in logs:
                f.write(f"Step: {state.global_step}, Eval Loss: {logs['eval_loss']}\n")
            f.flush()

        if state.global_step % int(args.save_steps) == 0 and state.best_model_checkpoint:
            checkpoint_dir = state.best_model_checkpoint or os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            trainable_params = {n: p.data for n, p in model.named_parameters() if p.requires_grad}
            torch.save(trainable_params, os.path.join(checkpoint_dir, "trainable_params.bin"))


# Fine-tuning setup
def fine_tune_model(model, tokenizer, dataset, save_dir, context_length, log_file_path):
    # Custom logging callback
    logging_callback = LoggingCallback(log_file_path)

    # Setup trainer
    trainer = SFTTrainer(
        dataset_text_field="messages",
        max_seq_length=context_length,
        tokenizer=tokenizer,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
        args=TrainingArguments(
            save_steps=20,
            logging_steps=1,
            num_train_epochs=2,
            output_dir=save_dir,
            evaluation_strategy="steps",
            do_eval=True,
            eval_steps=0.1,
            per_device_eval_batch_size=4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            log_level="debug",
            bf16=True,  # For Ampere GPUs
            weight_decay=0.01,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            learning_rate=1e-4,
            warmup_ratio=0.04,
            optim="adamw_torch",
        ),
        callbacks=[logging_callback]
    )

    # Disable cache for training
    model.config.use_cache = False

    trainer.train()


def main():
    # Constants
    model_id = "meta-llama/Meta-Llama-3.1-8B"
    dataset_name = "nenad1002/quantum_science_research_dataset"
    cache_dir = "./cache"
    save_dir = f"./results/{model_id.split('/')[-1]}_quantum_fine_tuned"
    context_length = 4096  # Adjust based on requirements
    log_file_path = os.path.join(cache_dir, "training_logs.txt")

    # Load dataset
    dataset = load_dataset(dataset_name)

    # Model and tokenizer setup
    bnb_config = create_bnb_config()
    model, tokenizer = load_model_and_tokenizer(model_id, cache_dir)

    # Ensure pad token is set up
    setup_pad_token(tokenizer, model)

    # Set specific parameters as trainable
    trainable_param_names = ["embed_tokens"]  # Adjust based on model type
    set_trainable_params(model, trainable_param_names)

    # Print a summary of the trainable parameters
    print_trainable_parameters_summary(model)

    # Fine-tune the model
    fine_tune_model(model, tokenizer, dataset, save_dir, context_length, log_file_path)


if __name__ == "__main__":
    main()
