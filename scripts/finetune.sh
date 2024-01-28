CUDA_VISIBLE_DEVICES=1 python ../finetune.py \
                            --model_path "/data/lypan/llms/bloom-560m" \
                            --neuron_info_path "../neuron_info/bloom-560m/" \
                            --log_path "../logs" \
                            --train_data_path "/data/lypan/peft/data/TED-TALKS-2020/" \
                            --lang_pairs en-ar en-de en-fr en-it en-zh ar-en de-en fr-en it-en zh-en \
                            --save_model_path "../finetune_results/finetune_agnos_bloom-560m_1"
