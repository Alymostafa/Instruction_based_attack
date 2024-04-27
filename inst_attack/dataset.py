from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
from dataset import * 
from utils import construct_meta_prompt, construction_prompts_scores
from config import *
from run import args

tokenizer_llama = AutoTokenizer.from_pretrained(llama_path)
optimization_task_info = f"""I have a question"""

def return_lens(seq_len):
    if seq_len == 500:
        return 167 , 333
    
    elif seq_len == 300:
        return 100 ,200
    
    elif seq_len == 200:
        return 66 ,134

def sort_df(df, max_length):

    df['all_text'] = df['prefix'] + df['suffix']

    a, b = extract_sos_prompt_response(df.all_text.tolist(), int(max_length))

    df['prefix'] = a
    df['suffix'] = b
    return df


def split_dataframe(df, subset_number, subset_size=335):
    start_idx = subset_number * subset_size
    if subset_number == (len(df) // subset_size):  # Check if it's the last subset
        df_subset = df.iloc[start_idx:]
    else:
        end_idx = (subset_number + 1) * subset_size
        df_subset = df.iloc[start_idx:end_idx]
    
    df_subset = df_subset.reset_index(drop=True)
    return df_subset
    
def extract_sos_prompt_response(random_samples, max_len):

  seq_len = int(args.max_length)
  
  pfx_len, sfx_len = return_lens(seq_len)

  idx_lst = []
  prompt_lst = []
  response_lst = []

  for i in tqdm(range(len(random_samples))):
    sentence_words = tokenizer_llama(random_samples[i])['input_ids']

    prompt = sentence_words[: seq_len-sfx_len]
    response = sentence_words[seq_len-sfx_len : seq_len]
    
    prompt_lst.append(tokenizer_llama.decode(prompt, skip_special_tokens=True))
    response_lst.append(tokenizer_llama.decode(response, skip_special_tokens=True))
    

  return prompt_lst, response_lst

def load_and_process_data(df, max_length, subset):

    df = sort_df(df, max_length)
    df = split_dataframe(df, subset, subset_size=335)
    return df



def load_data():
    supported_lengths = [200, 300, 500]
    supported_data_types = ['c4', 'github' ,'arxiv', 'cc', 'books']

    if args.max_length not in supported_lengths:
        raise ValueError(f"Current supported max_length for {args.data_type} dataset are: {supported_lengths}. Got: {args.max_length}")
    
    if args.data_type not in supported_data_types:
        raise ValueError(f"Current supported data types are: {supported_data_types}. Got: {args.data_type}")
        
    file_path = f"data/seq_{args.max_length}/{args.data_type}/{args.data_type}_{args.max_length}.csv"
    df = pd.read_csv(file_path)
    
    df = sort_df(df, args.max_length)
    df = split_dataframe(df, args.subset, subset_size=335)

    return df





def build_dataset(tokenizer, df, prompt_col, score_col):
    # load imdb with datasets
    ds = Dataset.from_pandas(df) 

    def tokenize(sample):
        previous_prompts_scores = construction_prompts_scores(sample[prompt_col], sample[score_col])
        meta_prompt_optimization = construct_meta_prompt(sorted(previous_prompts_scores), optimization_task_info)
        
        sample["input_ids"] = tokenizer.encode(meta_prompt_optimization)
        sample['inputs_masks'] = tokenizer(meta_prompt_optimization)
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def get_batches(dataset, batch_size=16):
    output_data = dict()
    dataset.set_format("pandas")
    output_data["query"] = dataset["query"].tolist()
    query_tensors = dataset["input_ids"].tolist()
    org_batch = dataset[org_gen_col].tolist()
    batches = [list(zip(query_tensors[i:i+batch_size], org_batch[i:i+batch_size])) for i in range(0, len(query_tensors), batch_size)]
    return batches

def tokenize_individual(prompt, score, gt, tokenizer):
    sample = {}
    previous_prompts_scores = construction_prompts_scores(prompt, score)
    meta_prompt_optimization = construct_meta_prompt(sorted(previous_prompts_scores), optimization_task_info)

    sample["input_ids"] = tokenizer.encode(meta_prompt_optimization)
    sample['inputs_masks'] = tokenizer(meta_prompt_optimization)
    sample["query"] = tokenizer.decode(sample["input_ids"])
    sample['gt']=gt
    return sample

def get_best_prompt(row):
    if row['rougel_init'] > row['opt_rougeL']:
        return row['init_prompt']
    else:
        return row['optimized_prompt']