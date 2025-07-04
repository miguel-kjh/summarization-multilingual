import os
import itertools

CONSTANTS = {
    "context_length": 16384,
    "quantization": False, 
    "wandb": True,
}

MODEL_NAMES = [
    "models/BSC-LT/salamandra-2b/english/lora/salamandra-2b-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-23-58-11",
    "models/BSC-LT/salamandra-2b/french/lora/salamandra-2b-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-20-37-53",
    "models/BSC-LT/salamandra-2b/canario/lora/salamandra-2b-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-01-50-39",
    "models/BSC-LT/salamandra-2b/italian/lora/salamandra-2b-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-21-36-58",
    "models/BSC-LT/salamandra-2b/german/lora/salamandra-2b-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-23-08-26",
    "models/BSC-LT/salamandra-2b/spanish/lora/salamandra-2b-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-00-52-14",
    "models/BSC-LT/salamandra-2b/portuguese/lora/salamandra-2b-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-19-40-54",
    "models/BSC-LT/salamandra-2b-instruct/english/lora/salamandra-2b-instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-02-41-48",
    "models/BSC-LT/salamandra-2b-instruct/french/lora/salamandra-2b-instruct-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-00-16-19",
    "models/BSC-LT/salamandra-2b-instruct/canario/lora/salamandra-2b-instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-04-18-33",
    "models/BSC-LT/salamandra-2b-instruct/italian/lora/salamandra-2b-instruct-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-00-56-39",
    "models/BSC-LT/salamandra-2b-instruct/german/lora/salamandra-2b-instruct-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-01-48-04",
    "models/BSC-LT/salamandra-2b-instruct/spanish/lora/salamandra-2b-instruct-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-03-33-14",
    "models/BSC-LT/salamandra-2b-instruct/portuguese/lora/salamandra-2b-instruct-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-11-23-39-44",
    "models/Qwen/Qwen2.5-0.5B/english/lora/Qwen2.5-0.5B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-22-25-34",
    "models/Qwen/Qwen2.5-0.5B/french/lora/Qwen2.5-0.5B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-12-06-01",
    "models/Qwen/Qwen2.5-0.5B/canario/lora/Qwen2.5-0.5B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-03-14-31",
    "models/Qwen/Qwen2.5-0.5B/italian/lora/Qwen2.5-0.5B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-03-33-03",
    "models/Qwen/Qwen2.5-0.5B/german/lora/Qwen2.5-0.5B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-13-09-42",
    "models/Qwen/Qwen2.5-0.5B/spanish/lora/Qwen2.5-0.5B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-02-49-32",
    "models/Qwen/Qwen2.5-0.5B/portuguese/lora/Qwen2.5-0.5B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-03-51-27",
    "models/Qwen/Qwen2.5-0.5B-Instruct/english/lora/Qwen2.5-0.5B-Instruct-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-13-47-05",
    "models/Qwen/Qwen2.5-0.5B-Instruct/french/lora/Qwen2.5-0.5B-Instruct-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-13-20-23",
    "models/Qwen/Qwen2.5-0.5B-Instruct/canario/lora/Qwen2.5-0.5B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-14-24-03",
    "models/Qwen/Qwen2.5-0.5B-Instruct/italian/lora/Qwen2.5-0.5B-Instruct-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-12-49-08",
    "models/Qwen/Qwen2.5-0.5B-Instruct/german/lora/Qwen2.5-0.5B-Instruct-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-13-34-27",
    "models/Qwen/Qwen2.5-0.5B-Instruct/spanish/lora/Qwen2.5-0.5B-Instruct-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-14-09-42",
    "models/Qwen/Qwen2.5-0.5B-Instruct/portuguese/lora/Qwen2.5-0.5B-Instruct-portuguese-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-13-05-21",
    "models/Qwen/Qwen2.5-1.5B/english/lora/Qwen2.5-1.5B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-18-37-59",
    "models/Qwen/Qwen2.5-1.5B/french/lora/Qwen2.5-1.5B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-04-56-14",
    "models/Qwen/Qwen2.5-1.5B/canario/lora/Qwen2.5-1.5B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-20-18-15",
    "models/Qwen/Qwen2.5-1.5B/italian/lora/Qwen2.5-1.5B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-17-06-57",
    "models/Qwen/Qwen2.5-1.5B/german/lora/Qwen2.5-1.5B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-17-51-20",
    "models/Qwen/Qwen2.5-1.5B/spanish/lora/Qwen2.5-1.5B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-19-32-24",
    "models/Qwen/Qwen2.5-1.5B/portuguese/lora/Qwen2.5-1.5B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-20-03-51-39",
    "models/Qwen/Qwen2.5-1.5B-Instruct/english/lora/Qwen2.5-1.5B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-19-00-52",
    "models/Qwen/Qwen2.5-1.5B-Instruct/french/lora/Qwen2.5-1.5B-Instruct-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-07-48-52",
    "models/Qwen/Qwen2.5-1.5B-Instruct/canario/lora/Qwen2.5-1.5B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-20-23-21",
    "models/Qwen/Qwen2.5-1.5B-Instruct/italian/lora/Qwen2.5-1.5B-Instruct-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-17-39-54",
    "models/Qwen/Qwen2.5-1.5B-Instruct/german/lora/Qwen2.5-1.5B-Instruct-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-18-19-11",
    "models/Qwen/Qwen2.5-1.5B-Instruct/spanish/lora/Qwen2.5-1.5B-Instruct-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-19-37-42",
    "models/Qwen/Qwen2.5-1.5B-Instruct/portuguese/lora/Qwen2.5-1.5B-Instruct-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-06-18-00",
    "models/Qwen/Qwen2.5-3B/english/lora/Qwen2.5-3B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-02-26-31",
    "models/Qwen/Qwen2.5-3B/french/lora/Qwen2.5-3B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-22-25-19",
    "models/Qwen/Qwen2.5-3B/canario/lora/Qwen2.5-3B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-05-34-26",
    "models/Qwen/Qwen2.5-3B/italian/lora/Qwen2.5-3B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-23-47-34",
    "models/Qwen/Qwen2.5-3B/german/lora/Qwen2.5-3B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-01-05-22",
    "models/Qwen/Qwen2.5-3B/spanish/lora/Qwen2.5-3B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-04-13-39",
    "models/Qwen/Qwen2.5-3B/portuguese/lora/Qwen2.5-3B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-21-09-08",
    "models/Qwen/Qwen2.5-3B-Instruct/english/lora/Qwen2.5-3B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-02-46-25",
    "models/Qwen/Qwen2.5-3B-Instruct/french/lora/Qwen2.5-3B-Instruct-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-22-35-54",
    "models/Qwen/Qwen2.5-3B-Instruct/canario/lora/Qwen2.5-3B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-09-50-52",
    "models/Qwen/Qwen2.5-3B-Instruct/italian/lora/Qwen2.5-3B-Instruct-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-20-23-56",
    "models/Qwen/Qwen2.5-3B-Instruct/german/lora/Qwen2.5-3B-Instruct-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-23-42-22",
    "models/Qwen/Qwen2.5-3B-Instruct/spanish/lora/Qwen2.5-3B-Instruct-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-06-44-48",
    "models/Qwen/Qwen2.5-3B-Instruct/portuguese/lora/Qwen2.5-3B-Instruct-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-12-21-12-19",
    "models/Qwen/Qwen3-0.6B/english/lora/Qwen3-0.6B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-04-13-10",
    "models/Qwen/Qwen3-0.6B/french/lora/Qwen3-0.6B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-12-57-03",
    "models/Qwen/Qwen3-0.6B/canario/lora/Qwen3-0.6B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-05-33-26",
    "models/Qwen/Qwen3-0.6B/italian/lora/Qwen3-0.6B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-16-40-25",
    "models/Qwen/Qwen3-0.6B/german/lora/Qwen3-0.6B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-16-48-38",
    "models/Qwen/Qwen3-0.6B/spanish/lora/Qwen3-0.6B-spanish-e2-b1-lr0.0002-wd0.0-c12000-peft-lora-r16-a32-d0.0-2025-06-14-12-07-40",
    "models/Qwen/Qwen3-0.6B/spanish/lora/Qwen3-0.6B-spanish-e2-b2-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-07-01-22-56-08",
    "models/Qwen/Qwen3-0.6B/portuguese/lora/Qwen3-0.6B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-06-13-34",
    "models/Qwen/Qwen3-0.6B-Base/english/lora/Qwen3-0.6B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-09-02-27",
    "models/Qwen/Qwen3-0.6B-Base/french/lora/Qwen3-0.6B-Base-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-07-22-06",
    "models/Qwen/Qwen3-0.6B-Base/canario/lora/Qwen3-0.6B-Base-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-10-22-11",
    "models/Qwen/Qwen3-0.6B-Base/italian/lora/Qwen3-0.6B-Base-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-07-56-40",
    "models/Qwen/Qwen3-0.6B-Base/german/lora/Qwen3-0.6B-Base-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-08-29-16",
    "models/Qwen/Qwen3-0.6B-Base/spanish/lora/Qwen3-0.6B-Base-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-09-47-02",
    "models/Qwen/Qwen3-0.6B-Base/portuguese/lora/Qwen3-0.6B-Base-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-06-49-20",
    "models/Qwen/Qwen3-1.7B/english/lora/Qwen3-1.7B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-19-16-53",
    "models/Qwen/Qwen3-1.7B/french/lora/Qwen3-1.7B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-08-37-05",
    "models/Qwen/Qwen3-1.7B/canario/lora/Qwen3-1.7B-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-20-59-42",
    "models/Qwen/Qwen3-1.7B/italian/lora/Qwen3-1.7B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-17-39-44",
    "models/Qwen/Qwen3-1.7B/german/lora/Qwen3-1.7B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-18-27-29",
    "models/Qwen/Qwen3-1.7B/spanish/lora/Qwen3-1.7B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-20-10-14",
    "models/Qwen/Qwen3-1.7B/portuguese/lora/Qwen3-1.7B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-07-01-50",
    "models/Qwen/Qwen3-1.7B-Base/english/lora/Qwen3-1.7B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-15-16-41",
    "models/Qwen/Qwen3-1.7B-Base/french/lora/Qwen3-1.7B-Base-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-12-29-52",
    "models/Qwen/Qwen3-1.7B-Base/canario/lora/Qwen3-1.7B-Base-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-17-15-40",
    "models/Qwen/Qwen3-1.7B-Base/italian/lora/Qwen3-1.7B-Base-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-13-25-22",
    "models/Qwen/Qwen3-1.7B-Base/german/lora/Qwen3-1.7B-Base-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-14-21-32",
    "models/Qwen/Qwen3-1.7B-Base/spanish/lora/Qwen3-1.7B-Base-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-16-19-21",
    "models/Qwen/Qwen3-1.7B-Base/portuguese/lora/Qwen3-1.7B-Base-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-11-12-39",
    "models/unsloth/Llama-3.2-1B/english/lora/Llama-3.2-1B-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-11-20-00",
    "models/unsloth/Llama-3.2-1B/french/lora/Llama-3.2-1B-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-10-25-07",
    "models/unsloth/Llama-3.2-1B/canario/lora/Llama-3.2-1B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-12-22-29",
    "models/unsloth/Llama-3.2-1B/italian/lora/Llama-3.2-1B-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-10-46-09",
    "models/unsloth/Llama-3.2-1B/german/lora/Llama-3.2-1B-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-23-10-37-25",
    "models/unsloth/Llama-3.2-1B/spanish/lora/Llama-3.2-1B-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-11-59-14",
    "models/unsloth/Llama-3.2-1B/portuguese/lora/Llama-3.2-1B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-09-30-28",
    "models/unsloth/Llama-3.2-3B/english/lora/Llama-3.2-3B-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-16-48-08",
    "models/unsloth/Llama-3.2-3B/french/lora/Llama-3.2-3B-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-14-28-40",
    "models/unsloth/Llama-3.2-3B/canario/lora/Llama-3.2-3B-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-18-28-10",
    "models/unsloth/Llama-3.2-3B/italian/lora/Llama-3.2-3B-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-15-23-13",
    "models/unsloth/Llama-3.2-3B/german/lora/Llama-3.2-3B-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-16-05-00",
    "models/unsloth/Llama-3.2-3B/spanish/lora/Llama-3.2-3B-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-17-43-51",
    "models/unsloth/Llama-3.2-3B/portuguese/lora/Llama-3.2-3B-portuguese-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-13-33-17",
    "models/unsloth/Llama-3.2-1B-Instruct/english/lora/Llama-3.2-1B-Instruct-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-16-28-29",
    "models/unsloth/Llama-3.2-1B-Instruct/french/lora/Llama-3.2-1B-Instruct-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-14-50-39",
    "models/unsloth/Llama-3.2-1B-Instruct/canario/lora/Llama-3.2-1B-Instruct-canario-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-17-37-13",
    "models/unsloth/Llama-3.2-1B-Instruct/italian/lora/Llama-3.2-1B-Instruct-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-15-23-56",
    "models/unsloth/Llama-3.2-1B-Instruct/german/lora/Llama-3.2-1B-Instruct-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-15-55-19",
    "models/unsloth/Llama-3.2-1B-Instruct/spanish/lora/Llama-3.2-1B-Instruct-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-17-04-30",
    "models/unsloth/Llama-3.2-1B-Instruct/portuguese/lora/Llama-3.2-1B-Instruct-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-14-19-54",
    "models/unsloth/Llama-3.2-3B-Instruct/english/lora/Llama-3.2-3B-Instruct-english-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-23-45-25",
    "models/unsloth/Llama-3.2-3B-Instruct/french/lora/Llama-3.2-3B-Instruct-french-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-21-40-38",
    "models/unsloth/Llama-3.2-3B-Instruct/canario/lora/Llama-3.2-3B-Instruct-canario-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-01-15-08",
    "models/unsloth/Llama-3.2-3B-Instruct/italian/lora/Llama-3.2-3B-Instruct-italian-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-22-23-51",
    "models/unsloth/Llama-3.2-3B-Instruct/german/lora/Llama-3.2-3B-Instruct-german-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-23-03-18",
    "models/unsloth/Llama-3.2-3B-Instruct/spanish/lora/Llama-3.2-3B-Instruct-spanish-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-17-00-33-26",
    "models/unsloth/Llama-3.2-3B-Instruct/portuguese/lora/Llama-3.2-3B-Instruct-portuguese-e1-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-19-42-53",
    "models/Qwen/Qwen3-4B/spanish/lora/Qwen3-4B-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-08-06-07",
    "models/Qwen/Qwen3-4B/english/lora/Qwen3-4B-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-04-08-35",
    "models/Qwen/Qwen3-4B/french/lora/Qwen3-4B-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-17-05-56",
    "models/Qwen/Qwen3-4B/german/lora/Qwen3-4B-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-15-01-02-28",
    "models/Qwen/Qwen3-4B/italian/lora/Qwen3-4B-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-16-10-42-20",
    "models/Qwen/Qwen3-4B/portuguese/lora/Qwen3-4B-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-14-21-54-40",
    "models/Qwen/Qwen3-4B-Base/english/lora/Qwen3-4B-Base-english-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-04-05-15",
    "models/Qwen/Qwen3-4B-Base/french/lora/Qwen3-4B-Base-french-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-21-44-02",
    "models/Qwen/Qwen3-4B-Base/german/lora/Qwen3-4B-Base-german-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-01-36-50",
    "models/Qwen/Qwen3-4B-Base/italian/lora/Qwen3-4B-Base-italian-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-23-41-59",
    "models/Qwen/Qwen3-4B-Base/portuguese/lora/Qwen3-4B-Base-portuguese-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-18-18-14-18",
    "models/Qwen/Qwen3-4B-Base/spanish/lora/Qwen3-4B-Base-spanish-e2-b1-lr0.0002-wd0.0-c8192-peft-lora-r16-a32-d0.0-2025-06-19-05-53-08",
]

DATASET_NAMES = [
    ("portuguese", "data/02-processed/portuguese"),
    ("french", "data/02-processed/french") ,
    ("italian", "data/02-processed/italian"),
    ("german", "data/02-processed/german"),
    ("english", "data/02-processed/english"),
    ("spanish", "data/02-processed/spanish"),
    #("canario", "data/02-processed/canario"),
]


def for_evaluation(model_name, dataset_name):
    """
    Generates generation scripts for different combinations of models, PEFT types, and datasets.
    Each script is saved in a specified output directory.
    """
    return f"""
    # Model architecture
    model_name="{model_name}"

    # Data
    dataset_name="{dataset_name}"

    
    python model_evaluate.py \\
    --model $model_name \\
    --verbose True \\
    --dataset $dataset_name \\
    --use_openai True \\
    --method "Truncate" \\
    --up False
"""

# Create an output directory for the scripts
output_dir = "scripts"
os.makedirs(output_dir, exist_ok=True)

# Generate a script for each combination of model, PEFT type, and dataset
indx = 0
for (model_name, dataset_name) in itertools.product(MODEL_NAMES, DATASET_NAMES):
    lang = dataset_name[0]
    if lang not in model_name:
        continue
    folder_data = dataset_name[1]
    simple_name = model_name.split("/")[2]
    script_filename = os.path.join(output_dir, f"generate_{indx+1}_{simple_name}_{lang}.sh")
    indx += 1
    
    bash_script = for_evaluation(
        model_name=model_name,
        dataset_name=folder_data,
    )

    # Save the script
    with open(script_filename, "w") as file:
        file.write(bash_script)

    # Make the script executable
    os.chmod(script_filename, 0o755)

    print(f"✅ Script '{script_filename}' generated.")

print("\n🚀 All scripts have been generated in the 'generated_scripts' folder.")

