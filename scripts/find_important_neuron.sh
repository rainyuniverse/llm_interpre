CUDA_VISIBLE_DEVICES=4,5 nohup python ../find_important_neuron.py \
                            --model_path "/data/lypan/llm_interpre/llms/Qwen1.5-7B" \
                            --save_folder_path "/data/lypan/llm_interpre/neuron_info/Qwen1.5-7b/" > ../find.log &