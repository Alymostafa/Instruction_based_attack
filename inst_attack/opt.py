import torch
from tqdm import tqdm
from utils import *
from scoring import get_scores
from run import args
from vllm import SamplingParams
from config import *

N_BEST_OF = int(args.no_n_sampling)
gen_kwargs = {"top_k": float(args.top_k), "top_p": float(args.top_p), "do_sample": True, 'temperature': float(args.temp)}
sampling_params = SamplingParams(max_tokens=int(args.max_length),top_k=1,top_p=1.0)

device = 'cuda'



def call_llm_for_optimization_non_batched(meta_prompt, reward_model, max_new_tokens=500, num_prompts=1):

    outputs = reward_model.generate(meta_prompt, sampling_params)        
    output_list = [output.outputs[0].text for output in outputs]
    return output_list



def self_refined(input_ids, gt, attacker_model, attacker_tokenizer, reward_model, beta_mem, gamma_overlap):
    response_tensors_best_of = []
    gen_len = 500

    query = torch.tensor(input_ids)
    # generating copies of the same query for the Best-of-n sampling

    queries = query.repeat((N_BEST_OF, 1))

    output = attacker_model.generate(queries.to(device), max_new_tokens=gen_len, **gen_kwargs).squeeze()

    decoded_tokens = attacker_tokenizer.batch_decode(output, skip_special_tokens = True)

    extracted_prompt = extract_gen_response(decoded_tokens)  
    
    response_tensors_best_of.append(extracted_prompt) 

    scores_best_of, extracted_prompts,all_text = get_scores(response_tensors_best_of,[gt], reward_model, beta_mem, gamma_overlap)

    best_score = [a.max().item() for a in scores_best_of]

    highest_prompt = [response_tensors_best_of[i][a.argmax().item()] for i, a in enumerate(scores_best_of)]

    extract_best_gen_respone = [extracted_prompts[i][a.argmax().item()] for i, a in enumerate(scores_best_of)]

    all_text = [all_text[i][a.argmax().item()] for i, a in enumerate(scores_best_of)]

    return highest_prompt[0] , best_score[0] ,extract_best_gen_respone[0],all_text[0], gt , response_tensors_best_of , scores_best_of
    


def get_generations(batches, attacker_model, attacker_tokenizer, reward_model, beta_mem, gamma_overlap):
    gt_list, response_tensors = [], []
    prompt_tensors_best_of ,best_score_list = [],[]
    global_all_text = []
    all_24_prompts = []
    all_24_score = []
    
    temp_list_opt_prompt = []
    temp_list_best_score = []
    temp_list_response = []
    temp_list_gt = []
    temp_all_text = []
    temp_24_prompts = []
    temp_24_score = []
    
    tqdm.pandas()
    for i in tqdm(range(len(batches))):
        gt = batches[i][0][1]
        input_ids = batches[i][0][0]        
#         ds_new = tokenize_individual(highest_prompt,best_score,gt)
        for iteration in range(args.num_iterations):
            print("iterations: ",iteration)

            highest_prompt , best_score ,extract_best_gen_respone,all_text, gt , prompts_24 , score_24 = self_refined(input_ids,
            gt, attacker_model, attacker_tokenizer, reward_model, beta_mem, gamma_overlap)

            # store 3 iteration per sample
            temp_24_prompts.append(prompts_24)
            temp_24_score.append(score_24)
            temp_all_text.append(all_text)
            temp_list_opt_prompt.append(highest_prompt)
            temp_list_best_score.append(best_score)
            temp_list_response.append(extract_best_gen_respone)
            temp_list_gt.append(gt)
            
            ds_new = tokenize_individual(highest_prompt,best_score,gt)
            input_ids,gt = ds_new['input_ids'] ,ds_new['gt']
        
        prompt_tensors_best_of.append(temp_list_opt_prompt)
        best_score_list.append(temp_list_best_score)
        gt_list.append(temp_list_gt)
        response_tensors.append(temp_list_response)
        global_all_text.append(temp_all_text)
        all_24_prompts.append(temp_24_prompts)
        all_24_score.append(temp_24_score)
        
        temp_list_opt_prompt = []
        temp_list_best_score = []
        temp_list_response = []
        temp_list_gt = []
        temp_24_prompts = []
        temp_24_score = []
        temp_all_text = []
        
    
    return prompt_tensors_best_of, best_score_list,response_tensors,gt_list ,global_all_text, all_24_prompts , all_24_score     



def compute_metric(dataset, scores_best_of, all_text_list, all_24_prompts, all_24_score):

    response_tensors_ref, response_tensors = [], []
    response_tensors_best_of = []

    output_data = {
        "gt_pre_training_sample": dataset[org_gen_col].tolist(),
        "init_prompt_score": dataset[score_col].tolist(),
        "init_pre_training_sample": dataset[init_gen_col].tolist(),
        "init_prompt": dataset[prompt_col].tolist(),
        "optimized_prompt": [response_tensors_best_of[i][a.index(max(a))] for i, a in enumerate(scores_best_of)],
        "optimized_pre_training_sample": [response_tensors[i][a.index(max(a))] for i, a in enumerate(scores_best_of)],
        "optimized_scores": [max(a) for a in scores_best_of],
        "optimized_pre_training_sample_all_text": [all_text_list[i][a.index(max(a))] for i, a in enumerate(scores_best_of)]
    }

    df_24_sample = pd.DataFrame({"all_prompt24": all_24_prompts, "all_score24": all_24_score})

    df_results = pd.DataFrame(output_data).dropna().reset_index(drop=True)

    df_results['rougel_init'] = 0
    df_results['precision'] = 0
    df_results['recall'] = 0
    df_results['f1'] = 0
    df_results['bertscore'] = 0
    df_results['init_bertscores'] = 0
    df_results['rouge1'] = 0
    df_results['rouge2'] = 0
    df_results['opt_rougeL'] = 0
    df_results['sacreblue'] = 0

    for i in range(len(df_results)):
        pred = df_results['optimized_pre_training_sample'][i]
        gt = df_results['gt_pre_training_sample'][i]

        init_pred = df_results['init_pre_training_sample'][i]

        opt_prompt = df_results['optimized_prompt'][i]

        precision, recall, f1 = f1_score_ngram(pred, gt, 3)

        bertscores = bertscore.compute(predictions=[pred], references=[gt], model_type=deberta_base_path, device='cuda')['f1'][0]

        init_bertscores = bertscore.compute(predictions=[init_pred], references=[gt], model_type=deberta_base_path, device='cuda')['f1'][0]

        rouge1, rouge2, rougel = get_rouge_scores(pred, gt)
        rouge1_init, rouge2_init, rougel_init = get_rouge_scores(init_pred, gt)

        sacreblue = sacrebleu.compute(predictions=[pred], references=[[gt]])['score']

        rougel_suffix_overlap = get_rouge_scores(opt_prompt, gt)[2]
        
        df_results.loc[i, 'suffix_overlap'] = rougel_suffix_overlap
        df_results.loc[i, 'precision'] = precision
        df_results.loc[i, 'recall'] = recall
        df_results.loc[i, 'f1'] = f1
        df_results.loc[i, 'bertscore'] = bertscores
        df_results.loc[i, 'init_bertscores'] = init_bertscores
        df_results.loc[i, 'rouge1'] = rouge1
        df_results.loc[i, 'rouge2'] = rouge2
        df_results.loc[i, 'opt_rougeL'] = rougel
        df_results.loc[i, 'sacreblue'] = sacreblue
        df_results.loc[i, 'rougel_init'] = rougel_init

    return df_results, df_24_sample