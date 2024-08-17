# Alpaca against Vicuna: Using LLMs to Uncover Memorization of LLMs
This is the official repository for "[Alpaca against Vicuna: Using LLMs to Uncover Memorization of LLMs](https://arxiv.org/pdf/2403.04801)"
![image](https://github.com/user-attachments/assets/55b07778-881d-476d-ad92-2bd8f877ddbe)

## Table of Contents

- [Installation](#installation)
- [Models](#models)
- [Data](#data)
- [Experiments](#experiments)
- [Hyperparameters](#hyperparameters)
- [Citation](#citation)


## Installation
To install the code's requirements, do the following:
```
pip install -r requirements.txt
```

## Models

You need to add the directories for models in the [config](https://github.com/Alymostafa/Instruction_based_attack/blob/main/inst_attack/config.py). For the Alpaca and Tulu models, you need to follow the 
instructions [here](https://github.com/allenai/open-instruct) to get the weight difference and obtain the final model. Also, you may need to add your wand API key to the config file.

## Data
After cloning the repo, you need to download and add the data files to the data folder. Please see [here](https://github.com/Alymostafa/Instruction_based_attack/tree/main/data) for instructions on downloading the data.

## Experiments
To run experiments, specify the arguments. For instance, to attack the ```OLMo``` model using the ```CC``` data subset with a sequence length of ```200```, the arguments would be:
```
python run.py --attacker 'zephyr' --victim 'olmo' --data_type 'cc' --max_length 200
```

## Hyperparameters

You can specify several hyperparameters; if you don't, it will be the default values used in the paper. Here are the descriptions of some of them:

- ```no_n_sampling```: number of best-of-n (rejection sampling) to be generated (increasing the value would increase the performance and runtime and vice-versa)
- ```model_size```: the size of the victim LLM, we provide 7b, 13b, and 30b for some models. Please check [here](https://github.com/Alymostafa/Instruction_based_attack/blob/main/inst_attack/models.py) for the provided sizes.
- ```data_type:``` We have five data subsets 'c4', 'github' ,'arxiv', 'cc', 'books.`
- ```max_length:``` we provide support for 200, 300, 500
- ```num_iterations:``` number of feedback loops, (increasing the value would increase the performance and runtime and vice-versa)
- ```objective```: we support two objectives, overlap, in which the initial prompt is constructed based on the full sequence(prefix+suffix), so you need to minimize the overlap between the generated instruction and suffix. The other value is no overlap, which means the initial prompts are built only on the prefix, so there is no need to minimize the overlap.
- ```subset:```: we split each data subset into three subsets for quicker runtime, each with a size of ~335. 

## Citation
If you find this useful in your research, please consider citing:
```
@article{kassem2024alpaca,
  title={Alpaca against Vicuna: Using LLMs to Uncover Memorization of LLMs},
  author={Kassem, Aly M and Mahmoud, Omar and Mireshghallah, Niloofar and Kim, Hyunwoo and Tsvetkov, Yulia and Choi, Yejin and Saad, Sherif and Rana, Santu},
  journal={arXiv preprint arXiv:2403.04801},
  year={2024}
}
```
