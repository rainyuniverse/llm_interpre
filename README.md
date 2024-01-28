# Usage

## Installation

```bash
git clone https://github.com/rainyuniverse/llm_interpre.git
cd llm_interpre
pip install -r requirements.txt
```

## Running

1. find the lang-agnostic neurons and lang-specific nerouns

```bash
cd scripts
bash find_important_neuron.sh
```

Remember to set the `--model_path` argument to the path of the LLMs and set the `--save_folder_path` argument to the path of the output folder.

2. finetune the LLMs using parallel data

```bash
bash finetune.sh
```

Parameter List:

- `--model_path`: the path of the LLMs you need to finetune
- `--neuron_info_path`: the path of neuron info file (last step output)
- `--log_path`: the path of the training log file
- `--train_data_path`: the path of the training data storage folder. Each language pair needs a separate folder.
- `--lang_pairs`: the language pairs you want to finetune, separated by space. According to our method, only one language pair can be entered at a time, e.g. fr-en.
- `--save_model_path`: the path of the finetuned model storage folder

3. evaluate the finetuned model

```bash
# generate the hypothesis data
python evaluate.py
cd scripts
# compute BLEU scores of the hypothesis data
bash evaluate.sh
```


