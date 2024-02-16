# finetune bloom
# CUDA_VISIBLE_DEVICES=4,5,6 nohup python ../finetune_bloom.py \
#                             --model_path "/data/lypan/llms/bloom-7b1" \
#                             --neuron_info_path "/data/lypan/llm_interpre/neuron_info/bloom-7b1/" \
#                             --log_path "../logs" \
#                             --train_data_path "/data/lypan/peft/data/TED-TALKS-2020/" \
#                             --lang_pairs en-ar en-de en-fr en-it en-zh ar-en de-en fr-en it-en zh-en \
#                             --save_model_path "/data/lypan/llm_interpre/finetune_results/bloom-7b1-ablation/specific" > ../finetune-speci.log &

# finetune llama
# CUDA_VISIBLE_DEVICES=3,4 python ../finetune_llama.py \
#                             --model_path "/data/lypan/llm_interpre/llms/llama-2-7b-hf" \
#                             --neuron_info_path "/data/lypan/llm_interpre/neuron_info/test/" \
#                             --log_path "../logs" \
#                             --train_data_path "/data/lypan/peft/data/TED-TALKS-2020/" \
#                             --lang_pairs en-ar en-de en-fr en-it en-zh ar-en de-en fr-en it-en zh-en \
#                             --save_model_path "/data/lypan/llm_interpre/finetune_results/fintune_llama2-7b"

# finetune Qwen
CUDA_VISIBLE_DEVICES=4,5,6 nohup python ../finetune_qwen.py \
                            --model_path "/data/lypan/llm_interpre/llms/Qwen1.5-7B" \
                            --neuron_info_path "/data/lypan/llm_interpre/neuron_info/Qwen1.5-7b/" \
                            --log_path "../logs" \
                            --train_data_path "/data/lypan/peft/data/TED-TALKS-2020/" \
                            --lang_pairs en-ar en-de en-fr en-it en-zh ar-en de-en fr-en it-en zh-en \
                            --save_model_path "/data/lypan/llm_interpre/finetune_results/qwen1.5-7b" > ../finetune.log &

