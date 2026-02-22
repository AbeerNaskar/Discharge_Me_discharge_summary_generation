#### fine tune code which take the data we have created using create_test_data_truncated.py (generate train valid, test data)


import time
import math

from codecarbon import EmissionsTracker



tracker = EmissionsTracker(
            project_name="fine-tune-truncated-prompt",  ## for carbon emission track
            output_dir="emissions_logs",  # Folder to store CSV logs
            measure_power_secs=1,         # Measurement frequency in seconds
            save_to_file=True
        )



def format_seconds(seconds):
    if seconds < 0:
        raise ValueError("Seconds cannot be negative")

    fractional, total_seconds = math.modf(seconds)
    total_seconds = int(total_seconds)

    milliseconds = round(fractional * 1000)

    if milliseconds == 1000:
        total_seconds += 1
        milliseconds = 0

    days, total_seconds = divmod(total_seconds, 86400)
    hours, total_seconds = divmod(total_seconds, 3600)
    minutes, seconds = divmod(total_seconds, 60)

    time_str = f"{seconds:02d}.{milliseconds:03d}s"
    if minutes > 0:
        time_str = f"{minutes:02d}m {time_str}"
    if hours > 0:
        time_str = f"{hours:02d}h {time_str}"
    if days > 0:
        time_str = f"{days}d {time_str}"

    return time_str





start_time = time.perf_counter()
print(f"[{format_seconds(0)}] Script started.")



import os, torch, pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel  # Unsloth's high‑level wrappers



max_seq_length = 8000 # Choose any! according to your need
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.


print('phi train truncated')


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "/path_to_unsloth_model/phi-4-unsloth-bnb-4bit", # other unsloth models: "phi-4-unsloth-bnb-4bit", "mistral-7b-instruct-v0.3-bnb-4bit", "Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    token = "", # # if required put your huggingface token
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)





####################################################################################
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict
from datasets import concatenate_datasets

PATH = '/path_to_data/'


df = pd.read_csv(PATH+"all_merged_data_train_truncated.csv") # converted train file
dataset = Dataset.from_pandas(df)


df = pd.read_csv(PATH+"all_merged_data_valid_truncated.csv") # converted valid file
eval_dataset = Dataset.from_pandas(df)


df = pd.read_csv(PATH+"all_merged_data_test_phase_2_truncated.csv") # converted test file
test2_dataset = Dataset.from_pandas(df)

###################################################################################








end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - start_time
print(f"[{format_seconds(elapsed_part1)}] model loaded.")
elapsed_part1 = end_part1






from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    eval_dataset=eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 50000,
        save_strategy="steps",
        save_total_limit = 1,
        eval_steps=50000,
        do_eval=True,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "./out_model/phi4",
        report_to = "none", 
    ),
)





tracker.start()


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")





trainer_stats = trainer.train()
eval_results = trainer.evaluate()  # you can skip this, this takes a long time




model.save_pretrained("./out_model/phi4/lora_model_truncated") # Local saving
tokenizer.save_pretrained("./out_model/phi4/lora_model_truncated")



#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")













end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - elapsed_part1
print(f"[{format_seconds(elapsed_part1)}] training done.")
elapsed_part1 = end_part1




emissions = tracker.stop()
print(f"Training: Estimated CO₂ emissions: {emissions:.6f} kg")




FastLanguageModel.for_inference(model) # Enable native 2x faster inference









tracker.start()




def summerize(inp):
    inputs = tokenizer([inp], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 1000, use_cache = True)
    out = tokenizer.batch_decode(outputs)[0].replace(inp, "").replace("<|im_end|>","")
    return out    


df = test2_dataset.to_pandas()

summ = [""]*len(df)




for i in range(len(df)):
    summ[i] = summerize(df['text'][i])
    if i%100==0:
        df['prediction'] = summ
        df.to_excel("test2_unsloth_phi4_as_it_is_truncated.xlsx", index = False) ## output file (generated result of test set)


df['prediction'] = summ
df.to_excel("test2_unsloth_phi4_as_it_is_truncated.xlsx", index = False) ## output file (generated result of test set)





end_part1 = time.perf_counter()
elapsed_part1 = end_part1 - elapsed_part1
print(f"[{format_seconds(elapsed_part1)}] test done.")



emissions = tracker.stop()
print(f"Testing: Estimated CO₂ emissions: {emissions:.6f} kg")
