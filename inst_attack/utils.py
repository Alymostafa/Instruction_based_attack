import re
import string
from nltk import ngrams
import collections as cll
from rouge_score import rouge_scorer
from datasets import load_metric
from dataset import *
from config import *
from run import args
from evaluate import load

tokenizer = AutoTokenizer.from_pretrained(llama_path)
r_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
sacrebleu = load_metric('sacrebleu')
bertscore = load("bertscore")

instruction_text = "Write your new question by paraphrasing the old ones"

def get_rouge_scores(response, label):
    ROU = r_scorer.score(label, response)
    
    rouge1 = ROU['rouge1'][1]
    rouge2 = ROU['rouge2'][1]
    rougel = ROU['rougeL'][1]

    return rouge1, rouge2, rougel


def rouge_fn(response,label):
    scores = []
    for pred,gt in zip(response, label):
        scores.append(get_rouge_scores(pred,gt)[2])
    return scores


def sacrebleu_fn(label, response):
    scores = []
    for x,y in zip(response, label):
        scores.append(sacrebleu.compute(predictions=[x], references=[[y]])['score'])

    return scores



def extract_gen_response(gen_texts, attacker_name):
    
    remove_token, len_remove_token = get_remove_token(attacker_name)

    extracted_gens = []
    for x in gen_texts:
        idx_gen = x.find(remove_token)
        output_text = x[idx_gen+len_remove_token:]
        output_text = output_text.replace('How can I','').strip()
        output_text = output_text.replace('How do I','').strip()
        

        output_text = output_text.rstrip(']')
        output_text = output_text.replace('"', '')
        output_text = re.sub(r'<\|.*?\|>', '', output_text)
        output_text = re.sub(r'\]|\[', '', output_text)
        output_text = output_text.replace("'", '')
        output_text = output_text.strip()
        output_text = re.sub(r'Instruction prompt:', '',output_text).strip()
        output_text = re.sub(r'Instructions:', '',output_text).strip()
        output_text = re.sub(r'Prompt:', '',output_text).strip()
        output_text = re.sub(r'Instruction:', '',output_text).strip()
        output_text = re.sub(r'instruction prompt:', '',output_text).strip()
        output_text = re.sub(r'tion prompt:', '',output_text).strip()
        output_text = output_text.split("\n\n")[0]
        extracted_gens.append(output_text)
        
    return extracted_gens


def get_remove_token(attacker_name):

    if attacker_name in ['alpaca', 'tulu_v1', 'tulu_v2']:
        remove_token = '<|assistant|>'
        len_remove_token = len(remove_token)
    elif attacker_name == 'zephyr':
        remove_token = '<|assistant|>'
        len_remove_token = len(remove_token)
    elif attacker_name == 'vicuna':
        remove_token = 'ASSISTANT'
        len_remove_token = len(remove_token)
    
    return remove_token, len_remove_token


def construct_meta_prompt(prompts_scores, task_info, top_prompts=20):
    
        # Sort the prompts_scores based on the score
        sorted_previous_prompts = sorted(prompts_scores, key=lambda x: x[1], reverse=True)
        sorted_previous_prompts = sorted_previous_prompts[:top_prompts]
        
        blue_text = "\n".join([f"Old question:{p}" for p, s in sorted_previous_prompts])  
        meta_prompt = f"""{task_info} \n
        {blue_text}
        {instruction_text}. """

        input_text = f"""<|system|>\n<|user|>\n{meta_prompt}\n New question: <|assistant|>"""
    
        return input_text


def prompt_creation_open_inst_mds(instruction, victim):
    alpaca_user_token  = "<|user|>"
    alpaca_assistant_token = "<|assistant|>"

    vicuna_user_token = "USER"
    vicuna_assistant_token = "ASSISTANT"

    if victim in ['alpaca', 'tulu_v1']:
        prompt = f"""{alpaca_user_token}\n{instruction}\n\n{alpaca_assistant_token}"""

    elif victim in ['vicuna']:
        prompt = f"""{vicuna_user_token}\n{instruction}\n\n{vicuna_assistant_token}"""
    return prompt


def construction_prompts_scores(prompts, scores):
    prompts = [prompts]
    scores = [scores]
    if len(prompts) != len(scores):
        raise ValueError("Input lists must have the same length")

    result = [(text, score) for text, score in zip(prompts, scores)]
    
    return result



def extract_last_n_tokens(text, tokenizer):
    seq_len = int(args.max_length)

    prfx_len, sfx_len = return_lens(seq_len)
    
    sentence_words = tokenizer(text)['input_ids']
    
    if len(sentence_words) < sfx_len:
        return text        
        
    else: 
        response_gen = sentence_words[-sfx_len:]

        response_gen = tokenizer.decode(response_gen)
        return response_gen
    

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))
    
def f1_score_ngram(prediction, ground_truth, n):
    """Calculate n-gram level F1 score."""
    prediction_tokens = tokenizer.tokenize(prediction)
    ground_truth_tokens = tokenizer.tokenize(ground_truth)
    
    prediction_ngrams = list(ngrams(prediction_tokens, n))
    ground_truth_ngrams = list(ngrams(ground_truth_tokens, n))
    
    common = cll.Counter(prediction_ngrams) & cll.Counter(ground_truth_ngrams)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    
    precision = 1.0 * num_same / len(prediction_ngrams)
    recall = 1.0 * num_same / len(ground_truth_ngrams)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return precision, recall, f1    