ori_data_prefix=/data/lypan/llm_interpre/data/flores200
trans_data_prefix=/data/lypan/llm_interpre/translation_results/template2/finetune_bloom-560m_3
# trans_data_prefix=/data/lypan/llm_interpre/translation_results/bloom-560m
# trans_data_prefix=/data/lypan/llm_interpre/translation_results/finetune_bloom-560m
lang_pairs=("en-ar" "en-de" "en-fr" "en-it" "en-zh"  "ar-en" "de-en" "fr-en" "it-en" "zh-en")
# lang_pairs=("fr-en")

declare -A lang_code_dict
lang_code_dict=( ["en"]="eng_Latn" ["ar"]="arb_Arab" ["de"]="deu_Latn" ["fr"]="fra_Latn" ["it"]="ita_Latn" ["zh"]="zho_Hans")

for lang_pair in "${lang_pairs[@]}"; do
    echo ${lang_pair}

    OLD_IFS=$IFS
    IFS="-"
    read -ra values <<< "$lang_pair"
    IFS=$OLD_IFS
    src=${values[0]}
    tgt=${values[1]}

    if [ "$tgt" == "zh" ]; then
        sacrebleu ${ori_data_prefix}/${lang_pair}/${lang_code_dict[$tgt]}.devtest -i ${trans_data_prefix}/${lang_pair}.txt -m bleu -b -w 4 -tok zh
    else
        sacrebleu ${ori_data_prefix}/${lang_pair}/${lang_code_dict[$tgt]}.devtest -i ${trans_data_prefix}/${lang_pair}.txt -m bleu -b -w 4
    fi
done
