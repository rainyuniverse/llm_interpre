from datasets import load_dataset
import os
import sys
import argparse
import deepspeed
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    default_data_collator,
    get_linear_schedule_with_warmup,
    set_seed,
    pipeline
)
import matplotlib.pyplot as plt
import numpy as np
parent_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_directory)
import torch
from tqdm import tqdm

output_dir = '/data/lypan/llms/bloom-7b1'
# tokenizer,model = llm_util.get_model('/home/sxy/Projects/Models/bloomz-7b1-mt')
tokenizer = AutoTokenizer.from_pretrained(output_dir, use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(output_dir, device_map="cuda:4", trust_remote_code=True)

def read_text(path):
    new_list = []
    with open(path,'r') as r:
        for i in r.readlines()[:500]:
            new_list.append(i.replace('\n',''))
    return new_list

ar = "/data/lypan/peft/data/flores200_dataset/dev/arb_Arab.dev"
en = "/data/lypan/peft/data/flores200_dataset/dev/eng_Latn.dev"
fr = "/data/lypan/peft/data/flores200_dataset/dev/fra_Latn.dev"
es = "/data/lypan/peft/data/flores200_dataset/dev/spa_Latn.dev"
zh = "/data/lypan/peft/data/flores200_dataset/dev/zho_Hans.dev"
de = "/data/lypan/peft/data/flores200_dataset/dev/deu_Latn.dev"
it = "/data/lypan/peft/data/flores200_dataset/dev/ita_Latn.dev"
ar_list = read_text(ar)
en_list = read_text(en)
fr_list = read_text(fr)
es_list = read_text(es)
zh_list = read_text(zh)
de_list = read_text(de)
it_list = read_text(it)


def get_prompt(language,e_q,q):
    if language == 'test':
        return "Translate the following sentence from English to French:\nEnglish:"+e_q+"\nFrench:"+q
    if language == 'Romanian':
        return "Translate the following sentence from English to Romanian:\nEnglish:"+e_q+"\nRomanian:"+q
    if language == 'Polish':
        return "Translate the following sentence from English to Polish:\nEnglish:"+e_q+"\nPolish:"+q
    if language == 'Hebrew':
        return "Translate the following sentence from English to Hebrew:\nEnglish:"+e_q+"\nHebrew:"+q
    if language == 'German':
        return "Translate the following sentence from English to German:\nEnglish:"+e_q+"\nGerman:"+q
    if language == 'Arabic':
        return "Translate the following sentence from English to Arabic:\nEnglish:"+e_q+"\nArabic:"+q
    if language == 'Arabic_r':
        return "Translate the following sentence from Arabic to English:\nArabic:"+q+"\nEnglish:"+e_q
    if language == 'Spanish':
        return "Translate the following sentence from English to Spanish:\nEnglish:"+e_q+"Spanish:"+q
    if language == 'Spanish_r':
        return "Translate the following sentence from Spanish to English:\nSpanish:"+q+"English:"+e_q
    if language == 'Chinese':
        return "Translate the following sentence from English to Chinese:\nEnglish:"+e_q+"Chinese:"+q
    if language == 'Chinese_r':
        return "Translate the following sentence from Chinese to English:\nChinese:"+q+"English:"+e_q
    if language == 'Persian':
        return "Translate the following sentence from English to Persian:\nEnglish:"+e_q+"Persian:"+q
    if language == 'Italian':
        return "Translate the following sentence from English to Italian:\nEnglish:"+e_q+"Italian:"+q
    if language == 'Dutch':
        return "Translate the following sentence from English to Dutch:\nEnglish:"+e_q+"Dutch:"+q
    if language == 'Portuguese':
        return "Translate the following sentence from English to Portuguese:\nEnglish:"+e_q+"Portuguese:"+q
    if language == 'Portuguese_r':
        return "Translate the following sentence from Portuguese to English:\nPortuguese:"+q+"English:"+e_q
    if language == 'Russian':
        return "Translate the following sentence from English to Russian:\nEnglish:"+e_q+"Russian:"+q
    if language == 'Slovenian':
        return "Translate the following sentence from English to Slovenian:\nEnglish:"+e_q+"Slovenian:"+q
    if language == 'Turkish':
        return "Translate the following sentence from English to Turkish:\nEnglish:"+e_q+"Turkish:"+q
    if language == 'French':
        return "Translate the following sentence from English to French:\nEnglish:"+e_q+"French:"+q
    if language == 'French_r':
        return "Translate the following sentence from French to English:\nFrench:"+q+"English:"+e_q
    if language == "in-context":
        return "Translate the following sentence from English to French:\nEnglish:from a long-term drive  to increase future freedom of action.\nFrench:d'une tendance sur le long terme à augmenter la liberté d'action future.\nEnglish:"+q+"\nFrench:"

    return "this is error"


en_texts = en_list
texts = zh_list
model.eval()
all_layer_activations = [[] for _ in range(model.config.num_hidden_layers)]

# 自定义forward hook函数
def get_activation(layer_idx):
    def hook(model, input, output):
        all_layer_activations[layer_idx].append(output[0].mean().item())
    return hook

# 注册hook
hooks = []
for i, layer in enumerate(model.model.layers):
    hook = layer.register_forward_hook(get_activation(i))
    hooks.append(hook)
# 对每个文本进行前向传播
for en_text, text in zip(en_texts, texts):
    q = get_prompt('Chinese', en_text, text)
    inputs = tokenizer(q, return_tensors="pt").to('cuda:4')
    with torch.no_grad():
        outputs = model(input_ids = inputs['input_ids'], attention_mask = inputs['attention_mask'])

# 移除hooks
for hook in hooks:
    hook.remove()

# 计算每层的平均激活值
avg_layer_activations = [sum(layer_activations) / len(layer_activations) for layer_activations in all_layer_activations]
# 计算每层的平均激活值的变化
delta_avg_layer_activations = [(avg_layer_activations[i+1] - avg_layer_activations[i]) for i in range(len(avg_layer_activations)-1)]

# 绘制每层的平均激活值
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
ax1.plot(avg_layer_activations, label='Average Activation')
ax1.legend()
ax2.plot(delta_avg_layer_activations, label='Delta Average Activation')
ax2.legend()
# plt.xlabel('Layer')
# plt.ylabel('Average Activation')
# plt.title('Average Activation per Layer in baichuan2-7b across 500 ar sentences')
plt.savefig('fig/qwen-zh.png', dpi = 300)