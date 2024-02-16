import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

import torch
from transformers import AutoTokenizer, BloomForCausalLM, AutoModelForCausalLM
from tqdm import tqdm
import json

lang_code_dict1 = {"ar": "arb_Arab", "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn", "zh": "zho_Hans"}


def read_data(file_path):
    sent_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for sent in file.readlines():
            sent_list.append(sent.replace("\n", ""))
    
    return sent_list

def read_translation_data(lang_direction):
    # folder_path = "/data/lypan/llm_interpre/data/flores200/"
    folder_path = "/data/lypan/llm_interpre/data/opus-test/"
    folder_path = folder_path + lang_direction + "/"
    src_lang, tgt_lang = lang_direction.split("-")

    # src_sent_list = read_data(folder_path + lang_code_dict1[src_lang] + ".devtest")
    # tgt_sent_list = read_data(folder_path + lang_code_dict1[src_lang] + ".devtest")
    src_sent_list = read_data(folder_path + "test." + src_lang)
    tgt_sent_list = read_data(folder_path + "test." + tgt_lang)

    return src_sent_list, tgt_sent_list

def get_prompt(src_sent, language_direction):
    # template2
    # template = 'How do you say \"{sent1}\" in {lang2}?'
    # template1
    # template = "{sent1} Say this using {lang2}"
    # template3
    template = "Translate \"{sent1}\" to {lang2}."
    source_language = language_direction.split("-")[0]
    target_language = language_direction.split("-")[1]

    prompt = template.format(sent1=src_sent, lang2=target_language)
    # print(prompt)

    # prompt = "Translate the sentence from " + source_language + " to " + target_language + ".\n"
    # prompt = prompt + source_language + ": " + src_sent + "\n"
    # prompt = prompt + "Corresponding " + target_language + " translation: "

    return prompt

def get_lang_agnos_speci_neuron():
    folder_path = "/data/lypan/llm_interpre/neuron_info/bloom-560m/"
    with open(folder_path + 'lang_agnos.json', 'r') as json_file:
        lang_agnos = json.load(json_file)
    with open(folder_path + 'lang_speci_by_lang.json', 'r') as json_file:
        lang_speci = json.load(json_file)

    return lang_agnos, lang_speci

lang_agnos, lang_speci = get_lang_agnos_speci_neuron()

# 根据 某一结构所有参数和神经元索引 寻找指定神经元参数：指定神经元参数不变，其他位置置0
def get_neuron_by_index(module_param, neuron_index):
    # 全零初始化weight和bias的mask矩阵
    mask_matrix = torch.zeros_like(module_param) if module_param is not None else None

    # 语言共有神经元位置置1，其他还是0
    mask_matrix[neuron_index] = 1

    # 只保留语言共有的神经元，其他神经元置零
    neuron_param = module_param * mask_matrix

    return neuron_param

# 根据当前batch语言对、指定结构、每个语言对的特定参数字典更新当前模型参数
def update_model_param(model, src_lang_code, tgt_lang_code, weight_name_list, bias_name_list, specific_param_dict):
    for name, param in model.named_parameters():
        if name in weight_name_list:
            module_name = name.split(".weight")[0]
            # 获取当前结构的语言共有神经元
            lang_agnos_param = get_neuron_by_index(module_param=param, neuron_index=lang_agnos[module_name])
            # 获取当前结构对应语言的语言特有神经元
            src_lang_speci_param = specific_param_dict[module_name][src_lang_code]["weight"]
            tgt_lang_speci_param = specific_param_dict[module_name][tgt_lang_code]["weight"]
            # 获取两种语言特有神经元的重叠部分
            overlap_lang_speci_param = specific_param_dict[module_name]["pretrain"]["weight"]
            # 将三种神经元加和，然后减去重叠的部分，得到目前语言对应对应的参数，并更新模型参数
            with torch.no_grad():
                param[:] = lang_agnos_param + src_lang_speci_param + tgt_lang_speci_param - overlap_lang_speci_param
        elif name in bias_name_list:
            module_name = name.split(".bias")[0]
            # 获取当前结构的语言共有神经元
            lang_agnos_param = get_neuron_by_index(module_param=param, neuron_index=lang_agnos[module_name])
            # 获取当前结构对应语言的语言特有神经元
            src_lang_speci_param = specific_param_dict[module_name][src_lang_code]["bias"]
            tgt_lang_speci_param = specific_param_dict[module_name][tgt_lang_code]["bias"]
            # 获取两种语言特有神经元的重叠部分
            overlap_lang_speci_param = specific_param_dict[module_name]["pretrain"]["bias"]
            # 将三种神经元加和，然后减去重叠的部分，得到目前语言对应对应的参数，并更新模型参数
            with torch.no_grad():
                param[:] = lang_agnos_param + src_lang_speci_param + tgt_lang_speci_param - overlap_lang_speci_param

def get_prompt_list(src_sent_list, language_direction):
    prompt_list = []
    for i in range(len(src_sent_list)):
        prompt_list.append(get_prompt(src_sent_list[i], language_direction))

    return prompt_list

def write_data(output_list, data_path):
    with open(data_path, "w", encoding="utf-8") as f:
        for line in output_list:
            f.write(line + "\n")

def get_batch_responses(tokenizer, model, text_batch, input_batch):
    output_list = []
    with torch.no_grad():
        outputs = model.generate(input_batch.input_ids, max_new_tokens=256)

    for i in range(len(input_batch.input_ids)):
        output_list.append(tokenizer.decode(outputs[i], skip_special_tokens=True))

    for i in range(len(output_list)):
        output_list[i] = output_list[i].replace(text_batch[i], "", 1).replace("\n", " ").replace("\r", " ")

    return output_list

if __name__ == "__main__":
    # 输出翻译结果保存路径
    output_path_prefix = "/data/lypan/llm_interpre/translation_results/template3/ablation/agnostic/"
    # 微调后的模型文件路径
    finetune_model_path = "/data/lypan/llm_interpre/finetune_results/bloom-7b1-ablation/agnostic/"

    tokenizer = AutoTokenizer.from_pretrained("/data/lypan/llms/bloom-7b1", use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(finetune_model_path, trust_remote_code=True).to("cuda")
    # specific_param_dict = torch.load(finetune_model_path + "specific_param_dict.pth")

    # module_name_list = list(specific_param_dict.keys())
    # # 对应module name列表的weight name和bias name列表
    # weight_name_list = [module_name + ".weight" for module_name in module_name_list]
    # bias_name_list = [module_name + ".bias" for module_name in module_name_list]

    lang_direction_list = [
        "en-ar", "en-de", "en-fr", "en-it", "en-zh", 
        "ar-en", "de-en", "fr-en", "it-en", "zh-en"
    ]
    lang_direction1_list = [
        "English-Arabic", "English-German", "English-French", "English-Italian", "English-Chinese",
        "Arabic-English", "German-English", "French-English", "Italian-English", "Chinese-English"
    ]

    batch_size = 4

    for i in range(len(lang_direction_list)):
        src_lang, tgt_lang = lang_direction_list[i].split("-")
        src_lang_code = lang_code_dict1[src_lang]
        tgt_lang_code = lang_code_dict1[tgt_lang]

        # 更新模型语言特有参数
        # update_model_param(model=model,
        #                    src_lang_code=src_lang_code,
        #                    tgt_lang_code=tgt_lang_code,
        #                    weight_name_list=weight_name_list,
        #                    bias_name_list=bias_name_list,
        #                    specific_param_dict=specific_param_dict)
        
        output_list = []
        src_sent_list, tgt_sent_list = read_translation_data(lang_direction_list[i])
        prompt_list = get_prompt_list(src_sent_list, lang_direction1_list[i])

        for j in tqdm(range(0, len(prompt_list), batch_size), desc=lang_direction_list[i]):
            inputs = tokenizer(prompt_list[j: j + batch_size], padding=True, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs.input_ids, max_new_tokens=128)

            for k in range(outputs.shape[0]):
                cur_output = tokenizer.decode(outputs[k], skip_special_tokens=True)
                cur_output = cur_output.replace(prompt_list[j + k], "", 1).replace("\n", " ").replace("\r", " ")
                output_list.append(cur_output)

        write_data(output_list, output_path_prefix + lang_direction_list[i] + ".txt")