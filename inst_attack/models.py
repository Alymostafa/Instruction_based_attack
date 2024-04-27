from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM
import torch
import re
from config import *
from run import args

def load_victim_model():
    
    ## Load Alpaca Model
    if args.victim == 'alpaca':
        if args.args.model_size == '7b':
            victim_dir = alpaca_7b_path
            reward_model = LLM(model=args.victim_dir,dtype="float16") 
            tensor_parallel_size=1
            gpu_memory_utilization=0.5

        elif args.model_size == '13b':
            victim_dir = alpaca_13b_path
            tensor_parallel_size=2
            gpu_memory_utilization=0.5
    
    ## Load Vicuna Model
    elif args.victim == 'vicuna':
        if args.model_size == '7b':
            victim_dir = llama_7b_path
            tensor_parallel_size=1
            gpu_memory_utilization=0.5

    ## Load Tulu Model
    elif args.victim == 'tulu_v1':
        if args.model_size == '7b':
            victim_dir = tulu_7b_path
            tensor_parallel_size=1
            gpu_memory_utilization=0.5
        
        elif args.model_size == '30b':
            victim_dir = tulu_30b_path
            tensor_parallel_size=4
            gpu_memory_utilization=0.7
    
    
    victim_model = LLM(model=victim_dir, dtype="float16", tensor_parallel_size=tensor_parallel_size, gpu_memory_utilization=gpu_memory_utilization) 
    victim_tokenizer = AutoTokenizer.from_pretrained(args.victim_dir)

    return victim_model, victim_tokenizer

def load_attacker_model():
    if args.attacker  == 'zephyr':
        attacker_dir = zephyr_path

    attacker_model = AutoModelForCausalLM.from_pretrained(
    attacker_dir,
    device_map="auto",torch_dtype=torch.float16)

    victim_tokenizer = AutoTokenizer.from_pretrained(attacker_dir)

    return attacker_model, victim_tokenizer




    
