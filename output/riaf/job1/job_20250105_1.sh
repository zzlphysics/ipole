#!/bin/bash
# SgrA*生成数据集, 包括偏振数据
# 定义日志文件名
LOG_FILE="job_$(date +%Y%m%d_%H%M%S).log"

# 在后台运行Python脚本，并将输出重定向到日志文件
nohup stdbuf -oL python job_20250105.py --run_count=10000 --csv_filename=output20250105 --h5file_dir=/root/hdd256/ipole/files_20250105 > "$LOG_FILE" 2>&1 &

# 输出提示信息
echo "Python脚本已在后台运行，日志文件为 $LOG_FILE"