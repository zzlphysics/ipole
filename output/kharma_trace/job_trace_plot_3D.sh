#! /bin/bash

file_dir="/home/zzl/Projects/ipole/output/kharma_trace/20250406_output/a05z00_trace_Rhigh1"

# 处理以_trace.h5结尾的文件
for file in $file_dir/*_trace.h5; do
    if [ -f "$file" ]; then
        echo "Processing $file"
        python trace_plot_3D.py --file_name $file
    fi
done
