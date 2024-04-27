chatglm2_path="./model"
#chatglm3_path="../chatglm3-6b/model"

CUDA_VISIBLE_DEVICES=0 python lora_classifier/run_lora_classifier.py \
    --stage sft \
    --model_name_or_path  ${chatglm2_path} \
    --do_train \
    --do_eval \
    --dataset  math23k\
    --dataset_dir ./data/math23k \
    --finetuning_type lora \
    --output_dir ./lora_classifier/saved/result_classifier \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --plot_loss True \
    --warmup_steps 6000 \
    --checkpoint_dir lora_classifier/saved/result_classifier \
    --fp16 True > ./lora_classifier/logs/run_classifier.log 2>&1