#!/bin/bash

# 设置工作目录
cd evaluation/datasets || exit 1

# 定义要排除的zip文件（用|分隔，用于grep -vE）
EXCLUDE="cifar-100.zip|mnist.zip"

# 并行数
PARALLEL_JOBS=8

# 导出函数以便parallel使用
extract_zip() {
    local zip_file="$1"
    echo "Starting: $zip_file"
    unzip "$zip_file" -x "train/*" -d . 
    echo "Finished: $zip_file"
}
export -f extract_zip

# 扫描所有zip文件，排除指定文件，并行解压
find all_zip -maxdepth 1 -name "*.zip" -type f | \
    grep -vE "$EXCLUDE" | \
    parallel -j "$PARALLEL_JOBS" --eta extract_zip {}

echo "All done!"