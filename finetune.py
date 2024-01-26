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

folder_path = "/data/lypan/llm_interpre/neuron_info/bloom-560m/"
lang_code_list = ['arb_Arab', 'fra_Latn', 'spa_Latn', 'eng_Latn', 'deu_Latn', 'ita_Latn', 'jpn_Jpan', 'rus_Cyrl', 'zho_Hans', 'zho_Hant']
lang_code_dict1 = {"ar": "arb_Arab", "en": "eng_Latn", "de": "deu_Latn", "fr": "fra_Latn", "it": "ita_Latn", "zh": "zho_Hans"}

# 翻译指令模板
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

with open(folder_path + 'lang_agnos.json', 'r') as json_file:
    lang_agnos = json.load(json_file)

with open(folder_path + 'lang_speci_by_lang.json', 'r') as json_file:
    lang_speci = json.load(json_file)

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
    # retain_index = torch.cat((
    #     lang_agnos_neuron_index, 
    #     src_lang_speci_neuron_index,
    #     tgt_lang_speci_neuron_index), dim=0).to(torch.int)

    # 2. specific only
    # retain_index = torch.cat((
    #     src_lang_speci_neuron_index,
    #     tgt_lang_speci_neuron_index), dim=0).to(torch.int)

    # 3. agnostic only
    retain_index = lang_agnos_neuron_index.to(torch.int)

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
    writer = SummaryWriter('./logs/agnos')

    module_name_list = [
        'transformer.h.3.input_layernorm', 'transformer.h.3.self_attention.query_key_value', 'transformer.h.3.self_attention.dense', 'transformer.h.3.post_attention_layernorm', 'transformer.h.3.mlp.dense_h_to_4h', 'transformer.h.3.mlp.dense_4h_to_h', 
        'transformer.h.4.input_layernorm', 'transformer.h.4.self_attention.query_key_value', 'transformer.h.4.self_attention.dense', 'transformer.h.4.post_attention_layernorm', 'transformer.h.4.mlp.dense_h_to_4h', 'transformer.h.4.mlp.dense_4h_to_h', 
        'transformer.h.21.input_layernorm', 'transformer.h.21.self_attention.query_key_value', 'transformer.h.21.self_attention.dense', 'transformer.h.21.post_attention_layernorm', 'transformer.h.21.mlp.dense_h_to_4h', 'transformer.h.21.mlp.dense_4h_to_h', 
        'transformer.h.22.input_layernorm', 'transformer.h.22.self_attention.query_key_value', 'transformer.h.22.self_attention.dense', 'transformer.h.22.post_attention_layernorm', 'transformer.h.22.mlp.dense_h_to_4h', 'transformer.h.22.mlp.dense_4h_to_h', 
    ]

    # 冻结指定结构之外的其他参数
    freeze_other_param(model, module_name_list)

    limit_len_per_lang = 100000
    data_path_prefix = "/data/lypan/peft/data/TED-TALKS-2020/"
    lang_direction = [
        "en-ar", "en-de", "en-fr", "en-it", "en-zh", 
        "ar-en", "de-en", "fr-en", "it-en", "zh-en"
    ]
    # lang_direction = ["zh-en"]
    lang_code_dict = {"ar": "Arabic", "en": "English", "de": "Germany", "fr": "French", "it": "Italian", "zh": "Chinese"}

    # 初始化多语言翻译指令字典
    # data_dict = {key: [] for key in lang_direction}
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
            # 运用分词器将指令进行分词 {"input_ids", "attention_mask", "labels"}
            # tokenized_instruction = process_func(instruction, tokenizer)
            # data_dict[lang_direction[i]].append(tokenized_instruction)
            data_list.append(instruction)

    # print(data_dict)

    # multilingual_dataset = MultilingualDataset(data_dict)
    # data_loader = DataLoader(multilingual_dataset, batch_sampler=batch_sampler)

    # for batch in data_loader:
    #     print(batch)

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

    # 假设每隔 update_frequency 个 batch 更新一次梯度
    update_frequency = 10
    # 在训练循环中使用一个计数器来跟踪更新的频率
    update_counter = 0
    last_all_mask_grad = None

    for epoch in range(num_epochs):
        dataloader_len = len(data_loader)
        pbar=tqdm(total=dataloader_len, desc=epoch)
        for batch_idx, batch in enumerate(data_loader):
            batch_lang_pairs = batch["lang_pair"]
            # print(batch_lang_pairs)
            src_lang, tgt_lang = batch_lang_pairs.split("-")
            src_lang_code = lang_code_dict1[src_lang]
            tgt_lang_code = lang_code_dict1[tgt_lang]

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
            # 反向传播
            loss.backward()

            weight_name_list = [module_name + ".weight" for module_name in module_name_list]
            bias_name_list = [module_name + ".bias" for module_name in module_name_list]

            # param.grad = mask_2(now_param.grad - mask_1(last_param.grad)) + mask_1(last_param.grad)
            # 通过减去前几个batch的累积梯度计算当前batch产生的梯度
            for name, param in model.named_parameters():
                if name in weight_name_list:
                    if last_all_mask_grad is not None:
                        param.grad = param.grad - last_all_mask_grad[name]
                elif name in bias_name_list:
                    if last_all_mask_grad is not None:
                        param.grad = param.grad - last_all_mask_grad[name]
                        # param.grad = all_mask_grad[name]

            # 对当前batch的梯度进行mask操作
            all_mask_grad = get_all_mask_grad(model, module_name_list, src_lang_code, tgt_lang_code)

            for name, param in model.named_parameters():
                if name in weight_name_list:
                    # 对当前batch产生的梯度进行mask
                    param.grad = all_mask_grad[name]
                    # 加上前几个batch进行mask操作后的梯度得到累积梯度
                    if last_all_mask_grad is not None:
                        param.grad = param.grad + last_all_mask_grad[name]
                elif name in bias_name_list:
                    param.grad = all_mask_grad[name]
                    if last_all_mask_grad is not None:
                        param.grad = param.grad + last_all_mask_grad[name]

            last_all_mask_grad = copy.deepcopy(all_mask_grad)

            # 判断是否达到更新频率
            if (batch_idx + 1) % update_frequency == 0:
                optimizer.step()
                optimizer.zero_grad()
                last_all_mask_grad = None

                update_counter += 1

                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{dataloader_len}], Update {update_counter}, Loss: {loss.item()}")
                writer.add_scalar("loss", loss.item(), update_counter)

            pbar.update(1)

        # 最后确保在训练结束时执行最后一次梯度更新
        if (batch_idx + 1) % update_frequency != 0:
            optimizer.step()
            optimizer.zero_grad()
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{dataloader_len}], Update {update_counter}, Loss: {loss.item()}")
            writer.add_scalar('loss', loss.item(), update_counter + 1)

        pbar.close()
        writer.close()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    model.save_pretrained("/data/lypan/llm_interpre/finetune_results/finetune_agnos_bloom-560m_1")