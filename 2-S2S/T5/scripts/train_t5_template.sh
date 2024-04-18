pretrained_model_path=../../model/t5-large


max_tokens=2048
gradient=128
batch_size=16


output_dir="scripts/model_t5/template-2class"
src_path=${output_dir}/src

mkdir -p $output_dir
mkdir -p $src_path

CUDA_VISIBLE_DEVICES=0 MKL_THREADING_LAYER=GNU python transformer/examples/pytorch/translation/run_maxtokens_translation.py \
    --model_name_or_path $pretrained_model_path \
    --do_train \
    --do_eval \
    --source_lang src \
    --target_lang tgt \
    --source_prefix "translate English to English: " \
    --train_file "../../data/wi+locness/template2seq_2class_t5/ABC.train.gold.bea19.json" \
    --validation_file "../../data/wi+locness/template2seq_2class_t5/ABCN.dev.gold.bea19.json" \
    --output_dir $output_dir \
    --per_device_train_batch_size $batch_size \
    --per_device_eval_batch_size 64 \
    --max_tokens_per_batch $max_tokens \
    --gradient_accumulation_steps $gradient \
    --max_source_length 128 \
    --max_target_length 128 \
    --max_steps 1400 \
    --learning_rate 0.001 \
    --num_beams 5 \
    --optim adafactor \
    --load_best_model_at_end \
    --save_strategy steps \
    --evaluation_strategy steps \
    --save_steps 200 \
    --eval_steps 200 \
    --overwrite_output_dir \
    --preprocessing_num_workers 30 \
    --logging_first_step \
    --logging_steps 50 \
    --predict_with_generate 2>&1 | tee -a ${src_path}/train.log
 

       
