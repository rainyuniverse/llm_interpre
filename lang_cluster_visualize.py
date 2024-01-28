from transformers import AutoTokenizer, BloomForCausalLM
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import copy
from find_important_neuron import find_all_target_modules, read_monolingual_data
import json
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse

def get_lang_agnos_speci_neuron(folder_path):
    with open(folder_path + 'lang_agnos.json', 'r') as json_file:
        lang_agnos = json.load(json_file)
    with open(folder_path + 'lang_speci.json', 'r') as json_file:
        lang_speci = json.load(json_file)

    return lang_agnos, lang_speci

def forward_hook(module, input, output, module_name):
    # 在前向传播时调用
    forward_cache.append(output)

def add_hooks(model, target_module_names):
    hook_forwards = []
    for name, module in model.named_modules():
        if name in target_module_names:
            print(name, module)
            handle_forward = module.register_forward_hook(lambda m, i, o, module_name=name: forward_hook(m, i, o, name))
            hook_forwards.append(handle_forward)

    return hook_forwards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='model path')
    parser.add_argument("--neuron_info_path", type=str, help="neuron info path")
    args = parser.parse_args()

    lang_code_list = ['arb_Arab', 'fra_Latn', 'spa_Latn', 'eng_Latn', 'deu_Latn', 'ita_Latn', 'jpn_Jpan', 'rus_Cyrl', 'zho_Hans', 'zho_Hant']

    model_path = args.model_path
    neuron_info_path = args.neuron_info_path
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    bloom = BloomForCausalLM.from_pretrained(model_path).to("cuda")

    target_module_name_list = find_all_target_modules(bloom)
    # 目标结构名称列表
    target_module_names = ["transformer.h.12.mlp.dense_h_to_4h"]

    language_agnostic_representation = {key: [] for key in target_module_names}
    language_specific_representation = {key: [] for key in target_module_names}

    forward_cache = []
    hook_forwards = add_hooks(bloom, target_module_names)

    repre_method = "part"

    language_agnostic_neurons, language_specific_neurons = get_lang_agnos_speci_neuron(neuron_info_path)

    for i in range(len(lang_code_list)):
        cur_lang_code = lang_code_list[i]

        sent_list = read_monolingual_data(cur_lang_code)

        for j in tqdm(range(len(sent_list))):
            text = sent_list[j]
            inputs = tokenizer(text, return_tensors="pt").to("cuda")

            input = inputs["input_ids"][:, 0:len(inputs["input_ids"][0]) - 1]
            label = inputs["input_ids"][:, 1:len(inputs["input_ids"][0])]
            outputs = bloom(input, labels=label)

            for k in range(len(target_module_names)):
                cur_module_name = target_module_names[k]

                if repre_method == "all":
                    agnostic_length = len(language_specific_neurons[cur_module_name])
                    specific_length = len(language_agnostic_neurons[cur_module_name])
                    all_indices = torch.arange(agnostic_length + specific_length)

                    agnostic_bool_mask_matrix = torch.isin(all_indices, torch.tensor(language_specific_neurons[cur_module_name]))
                    specific_bool_mask_matrix = ~agnostic_bool_mask_matrix

                    agnostic_mask_matrix = torch.where(agnostic_bool_mask_matrix, torch.tensor(0), torch.tensor(1))
                    specific_mask_matrix = torch.where(specific_bool_mask_matrix, torch.tensor(0), torch.tensor(1))
                    # print(agnostic_mask_matrix)

                    language_agnostic_representation[cur_module_name].append(forward_cache[k].detach().cpu() * agnostic_mask_matrix)
                    language_specific_representation[cur_module_name].append(forward_cache[k].detach().cpu() * specific_mask_matrix)
                    
                elif repre_method == "part":
                    language_agnostic_representation[cur_module_name].append(forward_cache[k].detach().cpu().index_select(
                        -1, torch.tensor(language_agnostic_neurons[target_module_names[k]])))
                    language_specific_representation[cur_module_name].append(forward_cache[k].detach().cpu().index_select(
                        -1, torch.tensor(language_specific_neurons[target_module_names[k]])))

            forward_cache = []

    mean_agnostic_repre = [tensor.mean(dim=1) for tensor in language_agnostic_representation["transformer.h.12.mlp.dense_h_to_4h"]]
    agnostic_repre = torch.stack(mean_agnostic_repre)
    mean_agnostic_repre = agnostic_repre.view(-1, agnostic_repre.shape[-1]) # [lang_num * sent_num, lang_agnos_neuron_num]

    mean_specific_repre = [tensor.mean(dim=1) for tensor in language_specific_representation["transformer.h.12.mlp.dense_h_to_4h"]]
    specific_repre = torch.stack(mean_specific_repre)
    mean_specific_repre = specific_repre.view(-1, specific_repre.shape[-1]) # [lang_num * sent_num, lang_speci_neuron_num]

    # 语言共有神经元表示聚类
    high_dimensional_data = mean_agnostic_repre

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    low_dimensional_data = tsne.fit_transform(high_dimensional_data)

    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#34495e', '#f1c40f', '#e67e22', '#e74c3c', '#5f27cd', '#95a5a6']
    scatter_list = []
    for i in range((int)(len(low_dimensional_data) / 997)):
        start = i * 997
        end = (i + 1) * 997
        scatter_list.append(plt.scatter(low_dimensional_data[start:end, 0], low_dimensional_data[start:end, 1], c=colors[i], s=2))
    plt.title('lang_agnos neuron t-SNE Visualization')
    plt.legend(scatter_list, lang_code_list)
    plt.savefig('fig/lang_agnos.png')

    # 语言特有神经元表示聚类
    high_dimensional_data = mean_specific_repre

    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    low_dimensional_data = tsne.fit_transform(high_dimensional_data)

    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#34495e', '#f1c40f', '#e67e22', '#e74c3c', '#5f27cd', '#95a5a6']
    scatter_list = []
    for i in range((int)(len(low_dimensional_data) / 997)):
        start = i * 997
        end = (i + 1) * 997
        scatter_list.append(plt.scatter(low_dimensional_data[start:end, 0], low_dimensional_data[start:end, 1], c=colors[i], s=2))
    plt.title('lang_speci neuron t-SNE Visualization')
    plt.legend(scatter_list, lang_code_list)
    plt.savefig('fig/lang_speci.png')