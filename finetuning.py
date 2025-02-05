import argparse
import os
from datasets import load_from_disk
from distutils.util import strtobool
import pandas as pd
import torch

from peft import LoraConfig, VeraConfig, LoKrConfig, LoHaConfig
from transformers import (
    TrainingArguments,
)
from trl import SFTTrainer

from evaluation.summary_generator import SummaryGenerator
from embedding_distilation.Connector import Connector
from utils import INSTRUCTION_TEMPLATE, SEED, create_model_and_tokenizer, setup_environment, generate_names_for_wandb_run, upload_to_wandb, wandb_end


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-14m")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--num_train_epochs", type=int, default=20)
    parse.add_argument("--lr", type=float, default=1e-4)
    parse.add_argument("--weight_decay", type=float, default=0.01)
    parse.add_argument("--context", type=int, default=512)
    parse.add_argument("--dataset_name", type=str, default="data/03-combined/tiny")
    parse.add_argument("--num_proc", type=int, default=10)
    parse.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--upload", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--neftune_noise_alpha", type=float, default=None) # https://arxiv.org/abs/2310.05914
    parse.add_argument("--output_dir", type=str, default="models")
    parse.add_argument("--packing", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parse.add_argument("--gradient_checkpointing", type=bool, default=True)
    parse.add_argument("--logging_steps", type=int, default=25)
    parse.add_argument("--eval_strategy", type=str, default="steps")
    parse.add_argument("--eval_steps", type=int, default=1)
    parse.add_argument("--push_to_hub", type=lambda x: bool(strtobool(x)), default=False)

    # conectors
    parse.add_argument("--type_connector", type=str, default=None)
    parse.add_argument("--connector",  type=str, default=None) 

    #loras parameters 
    # TODO: add differents adapters (Adapters, dora, etc)
    parse.add_argument("--peft_type", type=str, default="lora") # lora, dora, vera, loha, lokr, x-lora?

    parse.add_argument("--lora_r", type=int, default=16)
    parse.add_argument("--lora_alpha", type=int, default=32) # a trick use lora_r*2
    parse.add_argument("--lora_dropout", type=float, default=0.05)
    parse.add_argument("--lora_bias", type=str, default="none")
    parse.add_argument("--lora_task_type", type=str, default="CAUSAL_LM")
    parse.add_argument("--lora_target_modules", type=str, default="query_key_value,dense,dense_h_to_4h,dense_4h_to_h")
    # quantization
    parse.add_argument("--quantization", type=lambda x: bool(strtobool(x)), default=False)
    args = parse.parse_args()
    args.lora_target_modules = args.lora_target_modules.split(",")
    args.run_name = generate_names_for_wandb_run(args)
    dataset_name = args.dataset_name.split("/")[-1]
    peft = args.peft_type
    folder = os.path.join(args.output_dir, args.model_name_or_path, dataset_name, peft)
    os.makedirs(folder, exist_ok=True)
    args.output_dir = os.path.join(folder, args.run_name)
    return args

def create_dict_fo_peft_config(script_args) -> dict:
    return {
        "lora": LoraConfig(
            lora_alpha=script_args.lora_alpha,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=script_args.lora_target_modules,
        ),
        "dora": LoraConfig(
            lora_alpha=script_args.lora_alpha,
            use_dora=True,
            lora_dropout=script_args.lora_dropout,
            r=script_args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=script_args.lora_target_modules,
        ),
        "vera": VeraConfig(
            r=script_args.lora_r, 
            target_modules=script_args.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        "loha": LoHaConfig(
            r=script_args.lora_r,
            alpha=script_args.lora_alpha,
            target_modules=script_args.lora_target_modules,
            rank_dropout=script_args.lora_dropout,
            module_dropout=script_args.lora_dropout,
            task_type="CAUSAL_LM",
        ),
        "lokr": LoKrConfig(
            r=script_args.lora_r,
            alpha=script_args.lora_alpha,
            target_modules=script_args.lora_target_modules,
            rank_dropout=script_args.lora_dropout,
            module_dropout=script_args.lora_dropout,
            task_type="CAUSAL_LM",
        )
    }

if __name__ == "__main__":
    script_args = parse_args()
    setup_environment(script_args)

    ################
    # Model init kwargs & Tokenizer
    ################
    tokenizer, model = create_model_and_tokenizer(script_args)
    peft_hypers = create_dict_fo_peft_config(script_args)

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_name)

    ################
    # Training
    ################
    peft_config = None 
    if script_args.peft_type in peft_hypers:
        print(f"### {script_args.peft_type} Finetuning ###")
        peft_config = peft_hypers[script_args.peft_type]
    else:
        print("### Full Finetuning ###")

    training_arguments = TrainingArguments(
        per_device_train_batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        optim="paged_adamw_32bit",
        learning_rate=script_args.lr,
        fp16=True,
        max_grad_norm=0.3,
        num_train_epochs=script_args.num_train_epochs,
        evaluation_strategy=script_args.eval_strategy,
        eval_steps=script_args.eval_steps,
        warmup_ratio=0.05,
        group_by_length=True,
        output_dir=script_args.output_dir,
        report_to="wandb" if script_args.wandb else None,
        save_safetensors=True,
        lr_scheduler_type="cosine",
        seed=SEED,
        load_best_model_at_end=True,
        logging_strategy="steps",
        logging_steps=1,
        save_strategy=script_args.eval_strategy, # saving is done at the end of each epoch
    )    
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=script_args.context,
        packing=script_args.packing,
        processing_class=tokenizer,
        args=training_arguments,
    )

    summary_generator = SummaryGenerator(
        tokenizer, 
        script_args.device
    )

    #if script_args.wandb:
    #    initial_summary = summary_generator.generate_summaries(model, dataset["test"], num_samples=5)
    #    upload_to_wandb("Original Summaries", initial_summary)

    trainer.train(resume_from_checkpoint=None)
    trainer.save_model(script_args.output_dir)

    #test_summary = summary_generator.generate_summaries(trainer.model, dataset["test"], num_samples=len(dataset["test"]))
    #df_summary = pd.DataFrame(test_summary)
    #df_summary.to_excel(os.path.join(script_args.output_dir, "test_summary.xlsx"), index=False)

    #if script_args.wandb:
        # get 5 samples of generated summaries
    #    after_training_summary = summary_generator.generate_summaries(trainer.model, dataset["test"], num_samples=5)
    #    upload_to_wandb("Generated Summaries", after_training_summary)
        #wandb_end()



    # Save and push to hub
    #if script_args.push_to_hub:
    #    trainer.push_to_hub(dataset_name=script_args.dataset_name)