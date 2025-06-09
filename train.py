import argparse
import os
from datasets import load_from_disk
from distutils.util import strtobool
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer, SFTConfig
from unsloth import is_bfloat16_supported
from evaluation import summary_generator
from utils import CONTEXT_WINDOWS, SEED, count_trainable_params, setup_environment, generate_names_for_wandb_run, upload_to_wandb, wandb_end, PROJECT_NAME


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path")
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--num_train_epochs", type=int, default=2)
    parse.add_argument("--lr", type=float, default=2e-4)
    parse.add_argument("--weight_decay", type=float, default=0.01)
    parse.add_argument("--context", type=int, default=512)  # Context window size, adjust based on model 8198
    parse.add_argument("--dataset_name", type=str, default="data/02-processed/spanish")
    parse.add_argument("--num_proc", type=int, default=1)
    parse.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parse.add_argument("--wandb", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--upload", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--output_dir", type=str, default="models")
    parse.add_argument("--packing", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parse.add_argument("--gradient_checkpointing", type=bool, default=True)
    parse.add_argument("--logging_steps", type=int, default=25)
    parse.add_argument("--eval_strategy", type=str, default="steps")
    parse.add_argument("--eval_steps", type=int, default=25)
    parse.add_argument("--push_to_hub", type=lambda x: bool(strtobool(x)), default=False)
    parse.add_argument("--selection_strategy", type=str, default="random")  # random, top_k, top_p #TODO: Implement selection strategy (mAYBE NOT NEEDED)

    #loras parameters 
    parse.add_argument("--peft_type", type=str, default="lora") # lora, dora, vera, loha, lokr, x-lora?

    parse.add_argument("--lora_r", type=int, default=16)
    parse.add_argument("--lora_dropout", type=float, default=0.0)
    parse.add_argument("--lora_bias", type=str, default="none")
    parse.add_argument("--lora_target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parse.add_argument("--train_embeddings", type=lambda x: bool(strtobool(x)), default=False)
    # quantization
    parse.add_argument("--quantization", type=lambda x: bool(strtobool(x)), default=False)
    # is tiny dataset
    parse.add_argument("--tiny_dataset", type=lambda x: bool(strtobool(x)), default=False)
    args = parse.parse_args()
    args.lora_target_modules = args.lora_target_modules.split(",")
    args.run_name = generate_names_for_wandb_run(args)
    dataset_name = args.dataset_name.split("/")[-1]
    peft = args.peft_type
    folder = os.path.join(args.output_dir, args.model_name_or_path, dataset_name, peft)
    os.makedirs(folder, exist_ok=True)
    args.output_dir = os.path.join(folder, args.run_name)
    return args

def create_model_and_tokenizer(args):

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = args.model_name_or_path,
        max_seq_length = args.context,  # Context window size
        dtype = None,
        load_in_4bit = args.quantization, # quantization QLoRA 4-bit
    )
    return tokenizer, model


if __name__ == "__main__":
    script_args = parse_args()
    setup_environment(script_args)

    ################
    # Model init kwargs & Tokenizer
    ################
    tokenizer, model = create_model_and_tokenizer(script_args)

    # When adding special tokens
    target_modules   = script_args.lora_target_modules
    train_embeddings = script_args.train_embeddings

    if train_embeddings:
        target_modules = target_modules + ["lm_head"]

    model = FastLanguageModel.get_peft_model(
        model,
        r = script_args.lora_r,  # Rank of the adapters
        target_modules = target_modules,  # On which modules of the llm the lora weights are used
        lora_alpha = 2*script_args.lora_r, # scales the weights of the adapters (more influence on base model), 16 was recommended on reddit
        lora_dropout = script_args.lora_dropout, # Default on 0.05 in tutorial but unsloth says 0 is better
        bias = "none",    # "none" is optimized
        use_gradient_checkpointing = "unsloth", #"unsloth" for very long context, decreases vram
        random_state = SEED,  # Seed for reproducibility
        use_rslora = False,  # scales lora_alpha with 1/sqrt(r), huggingface says this works better
        loftq_config = None, # And LoftQ
    )

    ################
    # Dataset
    ################
    dataset = load_from_disk(script_args.dataset_name)
    if script_args.tiny_dataset:
        print("Using tiny dataset for testing")
        # select a small subset of the dataset for quick testing randomly
        dataset["train"] = dataset["train"].shuffle(seed=SEED).select(range(500))
        #dataset["validation"] = dataset["validation"].shuffle(seed=SEED).select(range(5))

    ################
    # Prepare dataset for training
    ################

    if tokenizer.chat_template:
        print("Using chat template for formatting prompts")
        def formatting_func(example):
            instruction = example["instruction"]
            empty_prompt = f"{instruction}\n{{document}}\n"
            messages = [
                {"role": "system", "content": example["system_prompt"]},
                {"role": "user", "content": empty_prompt.format(document=example["input"])},
                {"role": "assistant", "content": example["output"]}
            ]
            return tokenizer.apply_chat_template(messages, tokenize=False)
        dataset_train = dataset.map(lambda x: {"text": formatting_func(x)})
    else:
        EOS_TOKEN = tokenizer.eos_token
        def formatting_prompts_instruction(example):
            instruction = example["instruction"]
            empty_prompt = f"### Instruction:{instruction}\n### Input:{{document}}\n### Response:\n"
            training_prompts = []
            inference_prompts = []
            summaries = []
            for doc, sum in zip(example["input"] , example["output"]):
                inference_prompt = empty_prompt.format(document=doc)
                real_sum = sum.strip()
                training_prompt = inference_prompt + sum + EOS_TOKEN
                training_prompt = training_prompt.replace("\n", " ")  # Remove newlines for better tokenization
                training_prompt = training_prompt.strip()  # Remove leading/trailing spaces
                training_prompts.append(training_prompt)
                inference_prompts.append(inference_prompt)
                summaries.append(real_sum)

            return { "text" : training_prompts, }
        dataset_train = dataset.map(formatting_prompts_instruction, batched=True)

    ################
    # Training
    ################

    args = SFTConfig(
        per_device_train_batch_size = script_args.batch_size, # 8 is a good value for 24GB GPU
        gradient_accumulation_steps = 4, # process 4 batches before updating parameters (parameter update == step)
        num_train_epochs = script_args.num_train_epochs,
        learning_rate = script_args.lr,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        optim = "adamw_8bit", # "adamw_8bit" for 8-bit AdamW, "paged_adamw_32bit" for 32-bit AdamW with paging
        weight_decay = script_args.weight_decay,
        lr_scheduler_type = "cosine", # "cosine" for cosine decay, "linear" for linear decay
        warmup_ratio = 0.05, # 0.05 is a good value for warmup ratio
        seed = SEED,
        output_dir=script_args.output_dir,
        report_to="wandb" if script_args.wandb else None,
        logging_steps = 1,  
        eval_strategy="steps",
        save_strategy="epoch",            # Guarda un checkpoint al final de cada época
        eval_steps=script_args.eval_steps,
        max_grad_norm = 1.0, 
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset_train["train"],
        eval_dataset = dataset_train["validation"],
        formatting_func=lambda x: x["text"],
        dataset_text_field = "text",
        max_seq_length = script_args.context,
        dataset_num_proc = 2,
        args = args,
    )

    #if script_args.wandb:
    #    initial_summary = summary_generator.generate_summaries(model, dataset["test"], num_samples=1)
    #    upload_to_wandb("Original Summaries", initial_summary)

    torch.cuda.reset_peak_memory_stats()

    trainer_stats = trainer.train()

    peak_memory = torch.cuda.max_memory_allocated() / (1024**3)  # en GB

    print(f"Memoria máxima GPU usada: {peak_memory:.2f} GB")
    print("Tiempo total de entrenamiento:", trainer_stats.metrics["train_runtime"], "segundos")
    print("Velocidad:", trainer_stats.metrics["train_samples_per_second"], "ejemplos/segundo")

    trainable_params, total_params, percentage = count_trainable_params(model)
    print(f"Trainable parameters: {trainable_params:,} / Total parameters: {total_params:,} ({percentage:.2f}%)")

    ## clean the memory of GPU
    torch.cuda.empty_cache()

    if script_args.wandb:
        import wandb
        print("Save in WandB")
        stats_of_trainer = {
            "peak_memory (GB)": peak_memory,
            "train_runtime": trainer_stats.metrics["train_runtime"],
            "train_samples_per_second": trainer_stats.metrics["train_samples_per_second"],
            "trainable_params": trainable_params,
        }
        run_name = script_args.run_name
        
        # save the stats in wandb
        wandb.log(stats_of_trainer)

        print(f"WandB run initialized with name: {run_name}")
        wandb_end()


    ################
    # Save the model
    ################

    trainer.save_model(script_args.output_dir)
    print(f"Model saved to {script_args.output_dir}")

    #test_summary = summary_generator.generate_summaries(trainer.model, dataset["test"], num_samples=len(dataset["test"]))
    #df_summary = pd.DataFrame(test_summary)
    #df_summary.to_excel(os.path.join(script_args.output_dir, "test_summary.xlsx"), index=False)

    #if script_args.wandb:
        # get 5 samples of generated summaries
    #    after_training_summary = summary_generator.generate_summaries(trainer.model, dataset["test"], num_samples=5)
    #    upload_to_wandb("Generated Summaries", after_training_summary)
    #    wandb_end()



    # Save and push to hub
    #if script_args.push_to_hub:
    #    trainer.push_to_hub(dataset_name=script_args.dataset_name)