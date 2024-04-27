chatglm2_path="./model"
#chatglm3_path="../chatglm3-6b/model"
init_checkpoint="./lora/saved/result_infix_chatglm2_ori2"
#init_checkpoint="./lora_mt/saved/generator_2"
control_filed="on_DE_3_g5e-5_r6e-5_epoch10_test"



for iter in {1..1}
do
  if [ $iter -ne "1" ];
  then
    step=$(( iter-1 ))
    init_checkpoint="./lora_mt/saved/generator_${step}_all_${control_filed}"
  fi

#   update online
  echo "update online: predict stage"
#  CUDA_VISIBLE_DEVICES=0 python lora/run_lora.py \
#    --stage sft \
#    --model_name_or_path  ${chatglm2_path} \
#    --do_predict True\
#    --dataset  math23k\
#    --dataset_dir ./data/math23k \
#    --finetuning_type lora \
#    --output_dir ./lora_mt/cls_data \
#    --per_device_train_batch_size 4 \
#    --checkpoint_dir "${init_checkpoint}" \
#    --per_device_eval_batch_size 15 \
#    --max_samples 20 \
#    --predict_with_generate \
#    --fp16 True \
#      > ./lora_mt/logs/predict_${iter}_all_${control_filed}.log 2>&1


  PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
#  python ./lora_mt/update_train_cls.py


  PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本

  echo "classifier: training stage"
  CUDA_VISIBLE_DEVICES=0 python lora_classifier/run_lora_pcls.py \
    --stage sft \
    --model_name_or_path  ${chatglm2_path} \
    --do_train \
    --dataset  math23k \
    --dataset_dir ./lora_mt/cls_data \
    --finetuning_type lora \
    --output_dir ./lora_mt/saved/ranker_${iter}_all_${control_filed} \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 12 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --learning_rate 6e-5 \
    --num_train_epochs 10 \
    --plot_loss True \
    --warmup_steps 3000 \
    --checkpoint_dir ${init_checkpoint} \
    --fp16 True > ./lora_mt/logs/ranker_${iter}_all_${control_filed}.log 2>&1

  PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
#  init_checkpoint="./lora_mt/saved/ranker_${iter}"
  init_checkpoint="./lora_mt/saved/ranker_${iter}_all_${control_filed}"

  # train generator
  echo "generation: training stage"
  CUDA_VISIBLE_DEVICES=0 python lora/run_lora.py \
    --stage sft \
    --model_name_or_path  ${chatglm2_path} \
    --do_train \
    --do_eval True\
    --dataset  math23k\
    --dataset_dir ./data/math23k \
    --finetuning_type lora \
    --lora_alpha 32 \
    --output_dir ./lora_mt/saved/generator_${iter}_all_${control_filed} \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 6 \
    --lr_scheduler_type cosine \
    --logging_steps 1000 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --metric_for_best_model eval_accuracy \
    --learning_rate 5e-5 \
    --num_train_epochs 10 \
    --plot_loss True \
    --warmup_steps 3000 \
    --checkpoint_dir ${init_checkpoint} \
    --fp16 True > ./lora_mt/logs/generator_${iter}_all_${control_filed}.log 2>&1

  PID=$!; wait ${PID} # 等待前一个脚本执行完，再执行后面一个脚本
done
