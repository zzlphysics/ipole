#!/bin/bash

# 定义日志文件名
LOG_FILE="job_$(date +%Y%m%d_%H%M%S).log"

# 在后台运行Python脚本，并将输出重定向到日志文件
nohup stdbuf -oL python job_20241017.py --run_count=2100 --csv_filename=output20241017 --h5file_dir=files_20241017 > "$LOG_FILE" 2>&1 &

# 输出提示信息
echo "Python脚本已在后台运行，日志文件为 $LOG_FILE"