# finetune bloom
CUDA_VISIBLE_DEVICES=3 python ../finetune_bloom.py \
                            --model_path "/data/lypan/llm_interpre/llms/bloom-560m" \
                            --neuron_info_path "/data/lypan/llm_interpre/neuron_info/bloom-560m/" \
                            --log_path "../logs" \
                            --train_data_path "/data/lypan/peft/data/TED-TALKS-2020/" \
                            --lang_pairs en-ar \
                            --save_model_path "/data/lypan/llm_interpre/finetune_results/bloom-560m"

# finetune llama
# CUDA_VISIBLE_DEVICES=3,4 python ../finetune_llama.py \
#                             --model_path "/data/lypan/llm_interpre/llms/llama-2-7b-hf" \
#                             --neuron_info_path "/data/lypan/llm_interpre/neuron_info/test/" \
#                             --log_path "../logs" \
#                             --train_data_path "/data/lypan/peft/data/TED-TALKS-2020/" \
#                             --lang_pairs en-ar \
#                             --save_model_path "/data/lypan/llm_interpre/finetune_results/fintune_llama2-7b"
