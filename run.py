
import wandb
import os
import argparse

from inst_attack.config import *
from inst_attack.dataset import get_batches, build_dataset, load_data
from inst_attack.models import load_attacker_model, load_victim_model
from inst_attack.scoring import calculate_beta_gamma
from inst_attack.opt import compute_metric, get_generations
from inst_attack.dataset import get_best_prompt

os.environ['WANDB_API_KEY'] = wandb_api_key

parser = argparse.ArgumentParser(description='lm_extraction')
parser.add_argument('--attacker',  help='')
parser.add_argument('--victim',  help='')
parser.add_argument('--prompt',  default='new', help='')
parser.add_argument('--no_n_sampling',  help='')
parser.add_argument('--model_size',  default='7b', help='')
parser.add_argument('--data_type',  default='', help='')
parser.add_argument('--max_length', type=int,default=200, help='')
parser.add_argument('--temp', default=1.0, type=float, help='',)
parser.add_argument('--top_k', default=0.0, help='')
parser.add_argument('--top_p', default= 1.0, help='')
parser.add_argument('--output_dir',help='dir of the outputfile')
parser.add_argument('--num_iterations', default=3, type=int, help='')
parser.add_argument('--objective', default='overlap', help='')
parser.add_argument('--beta_mem', type=float, default=0.5, help='')
parser.add_argument('--gamma_overlap', type=float, default=0.5, help='')
parser.add_argument('--subset', type=int, default=1, help='')

global args
args = parser.parse_args()

config_sc = {
    "victim": str(args.victim),
    "prompt": str(args.prompt),    
    "attacker": str(args.attacker),
    "no_n_sampling": str(args.no_n_sampling),
    "temp": str(args.temp),
    "model_size": str(args.model_size),
    "data_type": str(args.data_type),
    "top_k": str(args.top_k),
    "top_p": str(args.top_p),
    "max_length":str(args.max_length),
    "beta_mem": str(args.beta_mem),
    "gamma_overlap": str(args.gamma_overlap),
    # "input_dir":str(args.input_dir),
    "subset": str(args.subset),
    
}


# Construct the experiment name based on the configuration parameters
exps_name = config_sc['victim'] + '_' + config_sc['prompt'] + '_' + config_sc['attacker'] + '_' +"no_"+ config_sc['no_n_sampling'] + '_' + "temp_"+config_sc['temp'] + '_' +"mz_"+ config_sc['model_size']+ '_' +"topk_"+ config_sc['top_k'] + '_' +"top_p"+ config_sc['top_p']+"_max_length_"+config_sc['max_length']+"_subset_"+str(config_sc['subset'])+"_"+str(config_sc['beta_mem'])+"_"+str(config_sc['gamma_overlap'])

# Construct the group name based on the victim model and model size
group_name = config_sc['victim'] + '_' + config_sc['model_size'] 

# Initialize wandb with the experiment name, group name, project, entity, and configuration parameters
wandb.init(name = exps_name, group = config_sc['data_type'] ,project = "instruction_based_attack", entity = "project", config = config_sc)


print("Loading Attacker model...")
attacker_model, attacker_tokenizer = load_attacker_model()
print("Attacker model loaded.")

print("Loading Victim model...")
victim_model, victim_tokenizer = load_victim_model()
print("Victim model loaded.")

print("Loading dataset...")
df = load_data()
print("Dataset loaded.")

beta_mem, gamma_overlap = calculate_beta_gamma()

dataset = build_dataset(attacker_tokenizer, df, prompt_col, score_col)
dataset.set_format("pandas")



batches = get_batches(dataset, batch_size=1)

print("Running attack...")
response_tensors_best_of, scores_best_of,response_tensors,gt_list ,all_text_list,all_24_prompts , all_24_score = get_generations(batches, attacker_model, attacker_tokenizer, victim_model, beta_mem, gamma_overlap)
print("Attack completed.")

df_results, df_24_sample = compute_metric(dataset, scores_best_of, all_text_list, all_24_prompts, all_24_score)

df_results = df_results.dropna()



df_results['best_rougeL'] = df_results.apply(lambda row: max(row['rougel_init'], row['opt_rougeL']), axis=1)
df_results['best_prompt'] = df_results.apply(get_best_prompt, axis=1)

print('Start logging...')
performance_table = wandb.Table(columns=["init_prompt_scores", "optimized_scores","best_rougeL"], data=[[df_results['init_prompt_score'].mean(), df_results['opt_rougeL'].mean(),df_results['best_rougeL'].mean()]])

metrics_performance_table = wandb.Table(columns=["rougel_init", "rouge_optimized", "precision", "recall",
                                                 "f1", "opt_bertscore", "non_opt_bertscore","rouge1", "rouge2",
                                                 "opt_rougeL", "sacreblue",'suffix_overlap','best_rougeL'],
                                                   data=[[df_results['init_prompt_score'].mean(), df_results['optimized_scores'].mean(), 
                                                          df_results['precision'].mean(), df_results['recall'].mean(),
                                                          df_results['f1'].mean(), df_results['bertscore'].mean(), df_results['init_bertscores'].mean(),
                                                          df_results['rouge1'].mean(), df_results['rouge2'].mean(), df_results['opt_rougeL'].mean(),
                                                           df_results['sacreblue'].mean(),df_results['suffix_overlap'].mean(),df_results['best_rougeL'].mean()]])


wandb.log({"performance_table": performance_table})

wandb.log({"metrics_performance_table": metrics_performance_table})

wandb.log({"test_samples": df_results})

print('Logging completed.')

df_24_sample.to_csv("results/final_Exp/"+config_sc['data_type']+'/prompts_scores/'+str(exps_name)+'.csv', index=False)
df_results.to_csv("results/final_Exp/"+config_sc['data_type']+'/'+str(exps_name)+'.csv', index=False)