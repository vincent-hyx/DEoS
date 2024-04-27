chatglm2_path="./model"
#chatglm3_path="../chatglm3-6b/model"
lr=6e-4
control_filed="run_${lr}"

for iter in {1..5}
do
  step=$(( iter-1 ))
  CUDA_VISIBLE_DEVICES=0 python lora/run_lora.py \
    --stage sft \
    --model_name_or_path  ${chatglm2_path} \
    --do_train \
    --do_eval True\
    --dataset  mawps-single-five-fold \
    --dataset_dir ./data/mawps-single-five-fold/fold${step} \
    --finetuning_type lora \
    --output_dir ./lora/mawps_slm/saved/result_mawps_slm_fold${step}_${control_filed} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --learning_rate ${lr} \
    --num_train_epochs 20 \
    --plot_loss True\
    --checkpoint_dir first_generate \
    --warmup_steps 500 \
    --fp16 True > ./lora/mawps_slm/result_mawps_slm_fold${step}_${control_filed}.log 2>&1

  PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
done