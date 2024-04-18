# checkpoint_path="scripts/model_t5/baseline"
checkpoint_path="scripts/model_t5/template-2class"
output_dir=${checkpoint_path}/results_trueTest


# test_file="../../data/wi+locness/template2seq_2class_t5/ABCN.dev.gold.bea19.json"
# test_file="../../data/dataWithPredictions/predicted/binary/test_template.json"
# test_file="../../data/dataWithPredictions/predicted/multi/test_template.json"
test_file="../../data/dataWithPredictions/predicted/trueTest/test_template_2class.json"

mkdir -p $output_dir

CUDA_VISIBLE_DEVICES=0 python transformer/examples/pytorch/translation/run_maxtokens_translation.py \
    --model_name_or_path $checkpoint_path \
    --do_predict \
    --source_lang src \
    --target_lang tgt \
    --per_device_eval_batch_size 8 \
    --max_tokens_per_batch 1024 \
    --source_prefix "translate English to English: " \
    --test_file $test_file \
    --output_dir $output_dir \
    --num_beams=5 \
    --overwrite_output_dir \
    --predict_with_generate \
    
