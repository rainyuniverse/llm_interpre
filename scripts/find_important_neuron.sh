CUDA_VISIBLE_DEVICES=3 nohup python ../find_important_neuron.py \
                            --model_path "/data/lypan/llm_interpre/llms/bloom-560m" \
                            --save_folder_path "/data/lypan/llm_interpre/neuron_info/test/" > ../find.log &