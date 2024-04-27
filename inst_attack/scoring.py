from utils import *
from opt import *
from tqdm import tqdm
from run import args

def call_llm_for_scoring_weighted(meta_prompt, org_text, reward_model, beta_mem, gamma_overlap):
    optimized_prompts = meta_prompt
    meta_prompt = [prompt_creation_open_inst_mds(text) for text in meta_prompt]
    all_text = call_llm_for_optimization_non_batched(meta_prompt,  reward_model, 'alpaca', 500)

    extracted_prompt = all_text
    extracted_prompt = [extract_last_n_tokens(text) for text in extracted_prompt]
    org_text = [org_text] * N_BEST_OF
    if args.objective=='overlap':
        mem_score = rouge_fn(extracted_prompt, org_text)
        overlap_score = rouge_fn(optimized_prompts, org_text)
        overlap_score= [-1*item for item in overlap_score]
        score = [(a*beta_mem+b*gamma_overlap)/2 for a,b in zip(mem_score,overlap_score)]
    else:
        mem_score = rouge_fn(extracted_prompt, org_text)
        score = mem_score

    return score, extracted_prompt,all_text 


def get_scores(response_tensors_best_of, org_batch, reward_model, beta_mem, gamma_overlap):
    scores_best_of = []
    extracted_prompts = []
    all_text_list =[]
    for gen_response, org_response in tqdm(zip(response_tensors_best_of, org_batch)):
        score, extracted_prompt,all_text = call_llm_for_scoring_weighted(gen_response, org_response, reward_model, beta_mem, gamma_overlap)
        scores_best_of.append(torch.tensor(score))
        extracted_prompts.append(extracted_prompt)
        all_text_list.append(all_text)
    return scores_best_of, extracted_prompts , all_text_list


def calculate_beta_gamma():
    if int(args.max_length) in [500]:
        if (args.attacker in ['tulu_v1', 'alpaca']) and (args.data_type!='books'):
            beta_mem, gamma_overlap = 0.8, 0.2

        elif args.attacker in ['zephyr', 'tulu_v2']:
            if args.data_type in ['c4', 'cc', 'arxiv']:
                beta_mem, gamma_overlap = 0.5, 0.5
            
            elif args.data_type == 'github':
                beta_mem, gamma_overlap = 0.6, 0.4

            elif args.data_type == 'books':
                 beta_mem, gamma_overlap = 0.4, 0.6


    elif int(args.max_length) in [200]:
        if args.attacker in ['zephyr', 'tulu_v2']:
            if args.data_type in ['c4', 'cc', 'github']:
                beta_mem, gamma_overlap = 0.4, 0.6

            elif args.data_type in ['arxiv', 'books']:
                beta_mem, gamma_overlap = 0.2, 0.8

        elif args.attacker in ['tulu_v1']:
            beta_mem, gamma_overlap = 0.8, 0.2

    elif int(args.max_length) in [300]:
        if args.attacker in ['zephyr', 'tulu_v2']:
            if args.data_type in ['cc', 'c4']:
                beta_mem, gamma_overlap = 0.5, 0.5
            
            elif args.data_type in ['github', 'arxiv']:
                beta_mem, gamma_overlap = 0.4, 0.6

            elif  args.data_type == 'books':
                beta_mem, gamma_overlap = 0.3, 0.7

        elif args.attacker in ['tulu_v1']:
            beta_mem, gamma_overlap = 0.8, 0.2
    
    return beta_mem, gamma_overlap