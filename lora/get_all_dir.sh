#folder="./lora/saved/result1026"
#
# softfiles=$(ls $folder)
# for sfile in ${softfiles}
# do
#      [ -d $sfile ] && echo "soft: ${sfile}"
#done
#!/bin/bash

# 指定路径
path="./lora/saved/result1026"

# 获取指定路径下的所有文件夹名称
directories=$(find "$path" -type d)
all_dir=()
# 遍历并输出文件夹名称
for dir in $directories; do
    # 提取文件夹名称
    dirname=$(basename "$dir")
    all_dir+=("$dirname")
done
sorted_dir=($(printf "%s\n" "${all_dir[@]}" | awk -F '-' '{print $2}' | sort -n | awk '{print "checkpoint-"$1}'))

sorted_dir=("${sorted_dir[@]:1}")

