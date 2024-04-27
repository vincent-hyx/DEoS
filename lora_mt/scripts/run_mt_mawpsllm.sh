chatglm2_path="./model"
#chatglm3_path="../chatglm3-6b/model"

#init_checkpoint="./lora_mt/saved/generator_2"

for i in {1..5}
do
  fold=$(( i-1 ))
  init_checkpoint="./lora/mawps/saved/result_mawps_fold${fold}_run_6e-4"
  for iter in {1..1}
  do
  control_filed="r5e-5_g5e-5"
  #   update online
    echo "update online: predict stage fold${fold}"
#    CUDA_VISIBLE_DEVICES=0 python lora/run_lora.py \
#      --stage sft \
#      --model_name_or_path  ${chatglm2_path} \
#      --do_predict True \
#      --dataset  cv_mawps \
#      --dataset_dir ./data/cv_mawps/fold${fold} \
#      --finetuning_type lora \
#      --output_dir ./lora_mt/cls_data/mawps/fold${fold} \
#      --per_device_train_batch_size 4 \
#      --checkpoint_dir "${init_checkpoint}" \
#      --per_device_eval_batch_size 15 \
#      --max_samples 20 \
#      --predict_with_generate \
#      --fp16 True \
#        > ./lora_mt/logs/mawps/fold${fold}/predict_fold${fold}_${control_filed}.log 2>&1


    PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
#    python ./lora_mt/update_train_cls.py ${fold}


    PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本

    echo "classifier: training stage fold${fold}"
#    CUDA_VISIBLE_DEVICES=0 python lora_classifier/run_lora_pcls.py \
#      --stage sft \
#      --model_name_or_path  ${chatglm2_path} \
#      --do_train \
#      --dataset  mawps \
#      --dataset_dir ./lora_mt/cls_data/mawps/fold${fold} \
#      --finetuning_type lora \
#      --output_dir ./lora_mt/saved/mawps/fold${fold}/ranker_${iter}_${control_filed} \
#      --per_device_train_batch_size 2 \
#      --per_device_eval_batch_size 1 \
#      --gradient_accumulation_steps 12 \
#      --lr_scheduler_type cosine \
#      --logging_steps 100 \
#      --evaluation_strategy epoch \
#      --save_strategy epoch \
#      --load_best_model_at_end True \
#      --metric_for_best_model eval_accuracy \
#      --learning_rate 5e-5 \
#      --num_train_epochs 5 \
#      --plot_loss True \
#      --warmup_steps 1000 \
#      --checkpoint_dir ${init_checkpoint} \
#      --fp16 True > ./lora_mt/logs/mawps/fold${fold}/ranker_${iter}_${control_filed}.log 2>&1

    PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
  #  init_checkpoint="./lora_mt/saved/ranker_${iter}"
    init_checkpoint="./lora_mt/saved/mawps/fold${fold}/ranker_${iter}_${control_filed}"
    control_filed="r5e-5_g3e-4"

    # train generator
    echo "generation: training stage fold${fold}"
    CUDA_VISIBLE_DEVICES=0 python lora/run_lora.py \
    --stage sft \
    --model_name_or_path  ${chatglm2_path} \
    --do_train \
    --do_eval True\
    --dataset  cv_mawps \
    --dataset_dir ./data/cv_mawps/fold${fold} \
    --finetuning_type lora \
    --output_dir ./lora_mt/saved/mawps/fold${fold}/generator_${iter}_${control_filed} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type cosine \
    --logging_steps 100 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --learning_rate 3e-4 \
    --num_train_epochs 10 \
    --plot_loss True\
    --checkpoint_dir ${init_checkpoint} \
    --warmup_steps 300 \
    --fp16 True > ./lora_mt/logs/mawps/fold${fold}/generator_${iter}_${control_filed}.log 2>&1

    PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
  done
done
