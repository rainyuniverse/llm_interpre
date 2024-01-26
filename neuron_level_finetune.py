import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from transformers import AutoTokenizer, BloomForCausalLM
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import random
from datasets import Dataset
from multilingual_dataset import MultilingualDataset, MultilingualBatchSampler, collate_fn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
import copy

lang_code_list = ["arb_Arab", "eng_Latn", "deu_Latn", "fra_Latn", "ita_Latn", "zho_Hans"]
lang_code_dict = {"ar": "arb_Arab", "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn", "zh": "zho_Hans"}
lang_code_dict1 = {"ar": "arb_Arab", "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn", "zh": "zho_Hans"}

translation_patterns = [
    ("How do you say \"{sent1}\" in {lang2}?", "{sent2}"),
    ("{sent1} How do you say this sentence in {lang2}?", "{sent2}"),
    ("{sent1} Say this using {lang2}", "{sent2}"),
    ("Translate from {lang1} to {lang2}:\n\n{sent1}", "{sent2}"),
    ("Translate \"{sent1}\" from {lang1} to {lang2}.", "{sent2}"),
    ("Translate \"{sent1}\" to {lang2}.", "{sent2}"),
    ("Translate the following.\n\n{lang1}: {sent1}\n\n{lang2}:", "{sent2}"),
    ("Translate the sentence from {lang1} to {lang2}.\n{lang1}: {sent1}\nCorresponding {lang2} translation: ", "{sent2}"),
    ("Translate the sentence from {lang1} to {lang2}.\n{lang1}: {sent1}\n{lang2}: ", "{sent2}"),
    ("Translate from {lang1} to {lang2}.\n{lang1}: {sent1}\n{lang2}: ", "{sent2}"),
]

def transform_pattern_to_instruction(lang_pair, replacements):
    pattern = random.choice(translation_patterns)
    data = {}
    data["lang_pair"] = lang_pair
    data["input"] = pattern[0].format(**replacements)
    data["output"] = pattern[1].format(**replacements)
    return data

def construct_replacements(sent1, sent2, lang1, lang2):
    replacements = {"sent1": sent1, "sent2": sent2, "lang1": lang1, "lang2": lang2}
    return replacements

def read_data(file_path):
    sent_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for sent in file.readlines():
            sent_list.append(sent.replace("\n", ""))
    
    return sent_list

def read_translation_data(src_file_path, tgt_file_path):
    src_sent_list = read_data(src_file_path)
    tgt_sent_list = read_data(tgt_file_path)

    return src_sent_list, tgt_sent_list

def process_func(example, tokenizer):
    MAX_LENGTH = 256

    instruction = example["input"].strip() + "\n"
    instruction = tokenizer(instruction)

    response = tokenizer(example["output"] + tokenizer.eos_token)

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {
        "lang_pair": example["lang_pair"],
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels":  labels
    }

def get_lang_agnos_speci_neuron():
    folder_path = "/data/lypan/llm_interpre/neuron_info/bloom-560m/"
    with open(folder_path + 'lang_agnos.json', 'r') as json_file:
        lang_agnos = json.load(json_file)
    with open(folder_path + 'lang_speci_by_lang.json', 'r') as json_file:
        lang_speci = json.load(json_file)

    return lang_agnos, lang_speci

lang_agnos, lang_speci = get_lang_agnos_speci_neuron()

def find_specific_param(model, module_name):
    weight_name = module_name + ".weight"
    bias_name = module_name + ".bias"
    param_dict = {"weight": None, "bias": None}
    for name, param in model.named_parameters():
        if name == weight_name:
            param_dict["weight"] = param
        elif name == bias_name:
            param_dict["bias"] = param

    return param_dict

# 根据 指定结构名称、语言共有神经元索引 寻找该结构的语言共有神经元：语言共有神经元部分参数不变，其他神经元参数置0
def get_lang_agnos_neuron_by_module_name(model, module_name, lang_agnos):
    lang_agnos_param_dict = {}
    lang_agnos_comple_param_dict = {}
    # 根据指定结构名称寻找对应结构的参数（param_dict的键包括weight和bias）
    param_dict = find_specific_param(model, module_name)
    # 根据指定结构名称寻找对应结构的语言共有神经元索引
    lang_agnos_index = lang_agnos[module_name]

    # 全零初始化weight和bias的mask矩阵
    mask_weight_matrix = torch.zeros_like(param_dict["weight"]) if param_dict["weight"] is not None else None
    mask_bias_matrix = torch.zeros_like(param_dict["bias"]) if param_dict["bias"] is not None else None
    # 语言共有神经元位置置1，其他还是0
    mask_weight_matrix[lang_agnos_index] = 1
    mask_bias_matrix[lang_agnos_index] = 1

    # 只保留语言共有的神经元，其他神经元置零
    lang_agnos_param_dict["weight"] = param_dict["weight"] * mask_weight_matrix
    lang_agnos_param_dict["bias"] = param_dict["bias"] * mask_bias_matrix

    # 计算语言共有神经元的补集
    lang_agnos_comple_param_dict["weight"] = param_dict["weight"] - lang_agnos_param_dict["weight"]
    lang_agnos_comple_param_dict["bias"] = param_dict["bias"] - lang_agnos_param_dict["bias"]
    
    return lang_agnos_param_dict, lang_agnos_comple_param_dict

# 根据 某一结构所有参数和神经元索引 寻找指定神经元参数：指定神经元参数不变，其他位置置0
def get_neuron_by_index(module_param, neuron_index):
    # 全零初始化weight和bias的mask矩阵
    mask_matrix = torch.zeros_like(module_param) if module_param is not None else None

    # 语言共有神经元位置置1，其他还是0
    mask_matrix[neuron_index] = 1

    # 只保留语言共有的神经元，其他神经元置零
    neuron_param = module_param * mask_matrix

    return neuron_param

# 获取当前指定结构中除当前语言的语言特定神经元的索引列表
def get_other_lang_specific_neuron_index(module_name, cur_lang):
    other_lang_specific_neuron_index = []
    for lang in lang_speci[module_name].keys():
        if lang != cur_lang:
            other_lang_specific_neuron_index.extend(lang_speci[module_name][lang])

    return other_lang_specific_neuron_index

# 对指定结构列表，初始化每个语言对的特定参数
def init_specific_param(model, module_name_list, lang_code_list):
    # 保存每个语言对初始特定参数的拷贝，由于每个语言对初始特定参数相同，所以只保存一份，保存到pretrain键对应的值下
    copy_lang_code_list = copy.deepcopy(lang_code_list)
    copy_lang_code_list.append("pretrain")

    specific_param_dict = {key: {key: {"weight": None, "bias": None} for key in copy_lang_code_list} for key in module_name_list}

    for module_name in module_name_list:
        for lang_code in copy_lang_code_list:
            lang_agnos_param_dict, lang_agnos_comple_param_dict = get_lang_agnos_neuron_by_module_name(model, module_name, lang_agnos)
            specific_param_dict[module_name][lang_code]["weight"] = lang_agnos_comple_param_dict["weight"].detach().clone()
            specific_param_dict[module_name][lang_code]["bias"] = lang_agnos_comple_param_dict["bias"].detach().clone()

    return specific_param_dict


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


def update_specific_param_dict(model, src_lang_code, tgt_lang_code, weight_name_list, bias_name_list, specific_param_dict):
    for name, param in model.named_parameters():
        if name in weight_name_list:
            module_name = name.split(".weight")[0]
            # 获取当前结构对应语言的语言特有神经元
            src_lang_speci_param = get_neuron_by_index(module_param=param, neuron_index=lang_speci[module_name][src_lang_code])
            tgt_lang_speci_param = get_neuron_by_index(module_param=param, neuron_index=lang_speci[module_name][tgt_lang_code])

            # 除了提取当前语言的特有神经元外，其他语言的特有神经元位置的值需要保持和预训练模型一致
            # 所以我们根据其他语言的特有神经元索引，从保存的预训练模型参数中提取其他语言的特有神经元的值
            src_supple_param = get_neuron_by_index(module_param=specific_param_dict[module_name]["pretrain"]["weight"], 
                                               neuron_index=get_other_lang_specific_neuron_index(module_name=module_name, cur_lang=src_lang_code))
            tgt_supple_param = get_neuron_by_index(module_param=specific_param_dict[module_name]["pretrain"]["weight"],
                                               neuron_index=get_other_lang_specific_neuron_index(module_name=module_name, cur_lang=tgt_lang_code))
            
            # 更新specific_param_dict中源语言和目标语言语言特有神经元的值
            specific_param_dict[module_name][src_lang_code]["weight"] = (src_lang_speci_param + src_supple_param).detach().clone()
            specific_param_dict[module_name][tgt_lang_code]["weight"] = (tgt_lang_speci_param + tgt_supple_param).detach().clone()
        elif name in bias_name_list:
            module_name = name.split(".bias")[0]

            src_lang_speci_param = get_neuron_by_index(module_param=param, neuron_index=lang_speci[module_name][src_lang_code])
            tgt_lang_speci_param = get_neuron_by_index(module_param=param, neuron_index=lang_speci[module_name][tgt_lang_code])

            src_supple_param = get_neuron_by_index(module_param=specific_param_dict[module_name]["pretrain"]["bias"], 
                                               neuron_index=get_other_lang_specific_neuron_index(module_name=module_name, cur_lang=src_lang_code))
            tgt_supple_param = get_neuron_by_index(module_param=specific_param_dict[module_name]["pretrain"]["bias"],
                                               neuron_index=get_other_lang_specific_neuron_index(module_name=module_name, cur_lang=tgt_lang_code))

            specific_param_dict[module_name][src_lang_code]["bias"] = (src_lang_speci_param + src_supple_param).detach().clone()
            specific_param_dict[module_name][tgt_lang_code]["bias"] = (src_lang_speci_param + src_supple_param).detach().clone()

def get_mask_grad(model, module_name, src_lang_code, tgt_lang_code):
    """返回单一指定结构的mask梯度

    Args:
        model (_type_): _description_
        module_name (_type_): _description_

    Returns:
        mask_grad: _description_
    """
    # 寻找对应结构的weight和bias参数
    param_dict = find_specific_param(model, module_name)
    weight_grad, bias_grad = None, None
    # 计算对应参数的梯度矩阵
    if param_dict["weight"] is not None:
        weight_grad = param_dict["weight"].grad
    if param_dict["bias"] is not None:
        bias_grad = param_dict["bias"].grad

    # 计算当前结构神经元数量
    # neuron_num = weight_grad.shape[0] if weight_grad is not None else bias_grad.shape[0]
    # 初始化weight和bias的mask矩阵
    mask_weight_matrix = torch.zeros_like(weight_grad) if weight_grad is not None else None
    mask_bias_matrix = torch.zeros_like(bias_grad) if bias_grad is not None else None

    lang_agnos_neuron_index = torch.tensor(lang_agnos[module_name])
    src_lang_speci_neuron_index = torch.tensor(lang_speci[module_name][src_lang_code])
    tgt_lang_speci_neuron_index = torch.tensor(lang_speci[module_name][tgt_lang_code])
    
    # 1. agnostic + specific
    retain_index = torch.cat((
        lang_agnos_neuron_index, 
        src_lang_speci_neuron_index,
        tgt_lang_speci_neuron_index), dim=0).to(torch.int)

    # 2. specific only
    # retain_index = torch.cat((
    #     src_lang_speci_neuron_index,
    #     tgt_lang_speci_neuron_index), dim=0).to(torch.int)

    # 3. agnostic only
    # retain_index = lang_agnos_neuron_index.to(torch.int)

    # 构造weight和bias的mask矩阵
    if mask_weight_matrix is not None:
        mask_weight_matrix[retain_index] = 1
    if mask_bias_matrix is not None:
        mask_bias_matrix[retain_index] = 1

    # 进行mask操作
    if weight_grad is not None:
        weight_grad = weight_grad * mask_weight_matrix
        # 将0的位置都替换为nan
        # weight_grad = torch.where(weight_grad == 0, torch.tensor(float('nan')), weight_grad)
    if bias_grad is not None:
        bias_grad = bias_grad * mask_bias_matrix
        # bias_grad = torch.where(bias_grad == 0, torch.tensor(float('nan')), bias_grad)

    mask_grad = {"mask_weight_grad": weight_grad, "mask_bias_grad": bias_grad}

    return mask_grad

def get_all_mask_grad(model, module_name_list, src_lang_code, tgt_lang_code):
    """返回指定结构列表的mask梯度

    Args:
        model (_type_): _description_
        module_name_list (_type_): _description_

    Returns:
        all_mask_grad: _description_
    """
    all_mask_grad = {}
    for i in range(len(module_name_list)):
        module_name = module_name_list[i]
        mask_grad = get_mask_grad(model, module_name, src_lang_code, tgt_lang_code)
        all_mask_grad[module_name + ".weight"] = mask_grad["mask_weight_grad"]
        all_mask_grad[module_name + ".bias"] = mask_grad["mask_bias_grad"]
    return all_mask_grad

def freeze_other_param(model, module_name_list):
    for name, param in model.named_parameters():
        if any(module_name in name for module_name in module_name_list):
            param.requires_grad = True
        else:
            param.requires_grad = False

if __name__ == "__main__":
    model_path = "/data/lypan/llms/bloom-560m"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BloomForCausalLM.from_pretrained(model_path, device_map="auto")
    writer = SummaryWriter('./logs/finetune-1-26')

    run_mode = "train"
    save_iter = 12500

    module_name_list = [
        'transformer.h.3.input_layernorm', 'transformer.h.3.self_attention.query_key_value', 'transformer.h.3.self_attention.dense', 'transformer.h.3.post_attention_layernorm', 'transformer.h.3.mlp.dense_h_to_4h', 'transformer.h.3.mlp.dense_4h_to_h', 
        'transformer.h.4.input_layernorm', 'transformer.h.4.self_attention.query_key_value', 'transformer.h.4.self_attention.dense', 'transformer.h.4.post_attention_layernorm', 'transformer.h.4.mlp.dense_h_to_4h', 'transformer.h.4.mlp.dense_4h_to_h', 
        'transformer.h.21.input_layernorm', 'transformer.h.21.self_attention.query_key_value', 'transformer.h.21.self_attention.dense', 'transformer.h.21.post_attention_layernorm', 'transformer.h.21.mlp.dense_h_to_4h', 'transformer.h.21.mlp.dense_4h_to_h', 
        'transformer.h.22.input_layernorm', 'transformer.h.22.self_attention.query_key_value', 'transformer.h.22.self_attention.dense', 'transformer.h.22.post_attention_layernorm', 'transformer.h.22.mlp.dense_h_to_4h', 'transformer.h.22.mlp.dense_4h_to_h', 
    ]

    # 对应module name列表的weight name和bias name列表
    weight_name_list = [module_name + ".weight" for module_name in module_name_list]
    bias_name_list = [module_name + ".bias" for module_name in module_name_list]

    # 冻结指定结构之外的其他参数
    freeze_other_param(model, module_name_list)
    # 初始化每个语言对的特定参数
    specific_param_dict = init_specific_param(model, module_name_list, lang_code_list)
    

    limit_len_per_lang = 100000
    data_path_prefix = "/data/lypan/peft/data/TED-TALKS-2020/"
    lang_direction = [
        "en-ar", "en-de", "en-fr", "en-it", "en-zh", 
        "ar-en", "de-en", "fr-en", "it-en", "zh-en"
    ]

    # 多语言翻译指令
    data_list= []

    for i in range(len(lang_direction)):
        src_lang, tgt_lang = lang_direction[i].split("-")

        src_data_path = data_path_prefix + lang_direction[i] + "/train." + src_lang
        tgt_data_path = data_path_prefix + lang_direction[i] + "/train." + tgt_lang

        src_sent_list, tgt_sent_list = read_translation_data(src_data_path, tgt_data_path)

        for j in tqdm(range(len(src_sent_list[0:limit_len_per_lang])), desc=lang_direction[i]):
            # 'sent1':'****', 'sent2':'****', 'lang1': '****', 'lang2': '****'
            replacements = construct_replacements(src_sent_list[j], tgt_sent_list[j], lang_code_dict[src_lang], lang_code_dict[tgt_lang])
            # 根据模板构建指令数据
            instruction = transform_pattern_to_instruction(lang_direction[i], replacements)
            data_list.append(instruction)

    dataset = Dataset.from_dict({
        "lang_pair": [sent_pair["lang_pair"] for sent_pair in data_list],
        "input": [sent_pair["input"] for sent_pair in data_list],
        "output": [sent_pair["output"] for sent_pair in data_list],
    })

    tokenized_ds = dataset.map(lambda example: process_func(example, tokenizer), remove_columns=dataset.column_names)
    batch_size = 8
    batch_sampler = MultilingualBatchSampler(tokenized_ds, batch_size)
    data_loader = DataLoader(tokenized_ds, batch_sampler=batch_sampler, collate_fn=collate_fn)

    learning_rate = 5e-5
    num_epochs = 1

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        dataloader_len = len(data_loader)
        pbar=tqdm(total=dataloader_len, desc=epoch)
        for batch_idx, batch in enumerate(data_loader):
            batch_lang_pairs = batch["lang_pair"]
            # print(batch_lang_pairs)
            src_lang, tgt_lang = batch_lang_pairs.split("-")
            src_lang_code = lang_code_dict1[src_lang]
            tgt_lang_code = lang_code_dict1[tgt_lang]
            
            # 测试模型能否能成功更新参数
            # for name, param in model.named_parameters():
            #     if name == "transformer.h.3.self_attention.dense.weight":
            #         with torch.no_grad():
            #             test_param = copy.deepcopy(param[7])

            # specific_param_dict["transformer.h.3.self_attention.dense"]["arb_Arab"]["weight"][7][0] = 0.1
            
            # 首先更新模型参数，更新语言特有参数
                        
            update_model_param(model=model,
                               src_lang_code=src_lang_code,
                               tgt_lang_code=tgt_lang_code,
                               weight_name_list=weight_name_list,
                               bias_name_list=bias_name_list,
                               specific_param_dict=specific_param_dict)

            # 测试模型能否能成功更新参数           
            # for name, param in model.named_parameters():
            #     if name == "transformer.h.3.self_attention.dense.weight":
            #         with torch.no_grad():
            #             test_param_1 = copy.deepcopy(param[7])

            batch_inputs_tensor_list = [torch.tensor(lst) for lst in batch["input_ids"]]
            batch_inputs_pad_tensor = pad_sequence(batch_inputs_tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")

            batch_labels_tensor_list = [torch.tensor(lst) for lst in batch["labels"]]
            batch_labels_pad_tensor = pad_sequence(batch_labels_tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id).to("cuda")
            
            batch_attention_mask_tensor_list = [torch.tensor(lst) for lst in batch["attention_mask"]]
            batch_attention_mask_pad_tensor = pad_sequence(batch_attention_mask_tensor_list, batch_first=True, padding_value=0).to("cuda")

            # 正向传播
            outputs = model(
                input_ids=batch_inputs_pad_tensor, 
                attention_mask=batch_attention_mask_pad_tensor, 
                labels=batch_labels_pad_tensor
            )
            loss = outputs.loss
            writer.add_scalar("loss", loss.item(), batch_idx)
            # 反向传播
            loss.backward()

            all_mask_grad = get_all_mask_grad(model, module_name_list, src_lang_code, tgt_lang_code)

            for name, param in model.named_parameters():
                if name in weight_name_list:
                    param.grad = all_mask_grad[name]
                elif name in bias_name_list:
                    param.grad = all_mask_grad[name]

            optimizer.step()
            optimizer.zero_grad()

            # 测试模型参数能否同步在specific_param_dict中更新
            if run_mode == "test":
                weight_copy = copy.deepcopy(specific_param_dict["transformer.h.3.self_attention.dense"]["arb_Arab"]["weight"])

            # 将模型参数同步在specific_param_dict中更新
            update_specific_param_dict(model=model, 
                                       src_lang_code=src_lang_code, 
                                       tgt_lang_code=tgt_lang_code, 
                                       weight_name_list=weight_name_list, 
                                       bias_name_list=bias_name_list, 
                                       specific_param_dict=specific_param_dict)
            
            # 测试模型参数能否同步在specific_param_dict中更新
            if run_mode == "test":
                weight_copy_1 = copy.deepcopy(specific_param_dict["transformer.h.3.self_attention.dense"]["arb_Arab"]["weight"])
                unequal_columns = (weight_copy != weight_copy_1).any(dim=1)
                print("不一致的列索引:", unequal_columns.nonzero())
                print(lang_speci["transformer.h.3.self_attention.dense"]["arb_Arab"])
                assert unequal_columns.nonzero().squeeze().tolist() == lang_speci["transformer.h.3.self_attention.dense"]["arb_Arab"]


            pbar.update(1)

            # 每隔save_iter保存一次模型
            if batch_idx % save_iter == 0 and batch_idx != 0:
                folder_path = "/data/lypan/llm_interpre/finetune_results/finetune_bloom-560m_3/" + str(batch_idx)
                os.makedirs(folder_path, exist_ok=True)
                model.save_pretrained(folder_path + "/")
                torch.save(specific_param_dict, folder_path + "/specific_param_dict.pth")


        pbar.close()

    # 模型保存
    folder_path = "/data/lypan/llm_interpre/finetune_results/finetune_bloom-560m_3/last"
    os.makedirs(folder_path, exist_ok=True)
    model.save_pretrained(folder_path + "/")
    torch.save(specific_param_dict, folder_path + "/specific_param_dict.pth")