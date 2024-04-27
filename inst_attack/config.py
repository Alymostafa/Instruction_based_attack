
# This file contains configuration settings for an instruction-based attack.

# Paths to different language models
llama_path = "/huggyllama/llama-7b"  # Path to the Llama-7b language model
alpaca_7b_path = "allenai/open-instruct-stanford-alpaca-7b"  # Path to the Alpaca-7b language model
alpaca_13b_path = "allenai/open-instruct-stanford-alpaca-13b"  # Path to the Alpaca-13b language model
llama_7b_path = "lmsys/vicuna-7b-v1.3"  # Path to the Vicuna-7b language model
tulu_7b_path = "allenai/tulu-7b"  # Path to the Tulu-7b language model
tulu_30b_path = "allenai/tulu-30b"  # Path to the Tulu-30b language model
zephyr_path = "HuggingFaceH4/zephyr-7b-beta"  # Path to the Zephyr-7b language model
deberta_base_path = "microsoft/deberta-base"  # Path to the DeBERTa-base language model

# Wandb API key
wandb_api_key = "1234"

# Column names in the dataset
prompt_col = 'question'  # Name of the column containing the prompts/questions
score_col = 'rougel'  # Name of the column containing the scores/rouge-l values
org_gen_col = 'suffix'  # Name of the column containing the original generated responses
init_gen_col = 'gen_response'  # Name of the column containing the initial generated responses