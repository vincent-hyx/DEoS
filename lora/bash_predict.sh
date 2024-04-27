path="./lora/saved/result_infix_chatglm2_ori2"
chatglm3_path="../chatglm3-6b/model/"
chatglm2_path="./model"
# 获取指定路径下的所有文件夹名称
directories=$(find "$path" -name "checkpoint-*" -type d)
all_dir=()
# 遍历并输出文件夹名称
for dir in $directories; do
    # 提取文件夹名称
    dirname=$(basename "$dir")
    all_dir+=("$dirname")
done
sorted_dir=($(printf "%s\n" "${all_dir[@]}" | awk -F '-' '{print $2}' | sort -n | awk '{print "checkpoint-"$1}'))

sorted_dir=("${sorted_dir[@]:1}")

# 遍历数组并拼接字符串
for element in "${sorted_dir[@]}"; do
    result="$result $element"
done

result="$path$result"

# control whether or not to test the best model
test_best=True
if [ $test_best == True ];
then
  result="${path}"
fi


CUDA_VISIBLE_DEVICES=0 python lora/run_lora.py \
    --stage sft \
    --model_name_or_path  ${chatglm2_path} \
    --do_predict True\
    --dataset  math23k\
    --dataset_dir ./data/math23k \
    --finetuning_type lora \
    --output_dir ./lora_mt/cls_data \
    --per_device_train_batch_size 4 \
    --checkpoint_dir "${result}" \
    --per_device_eval_batch_size 15 \
    --max_samples 20 \
    --predict_with_generate \
    --fp16 True \
      > ./lora/predict.log 2>&1
