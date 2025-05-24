#! /bin/bash

export OMP_NUM_THREADS=16

# spack env activate ipole

RHIGH_LIST=( 160 )



# FILE_DIR="/home/zzl/QH-918/DataSet/Project/kharma_data/kharma-data-sync/20241217_cuda_kz_sane_a05000_eta+05_288-192-128_6-12_rout1000_min020_gamma13"
# OUTPUT_NAME="a05ez5_trace"

# FILE_DIR="/home/zzl/QH-918/DataSet/Project/kharma_data/kharma-data-sync/20241217_cuda_kz_sane_a09375_eta-01_288-192-128_6-12_rout1000_min020_gamma13"
# OUTPUT_NAME="a09ef1_trace"

# FILE_DIR="/home/zzl/QH-918/DataSet/Project/kharma_data/kharma-data-sync/20241217_cuda_sane_a09375_288-192-128_6-12_rout1000_min020_gamma13"
# OUTPUT_NAME="a09e00_trace"

FILE_DIR="/home/zzl/QH-918/DataSet/Project/kharma_data/kharma-data-sync/20241217_cuda_kz_sane_a09375_eta+05_288-192-128_6-12_rout1000_min020_gamma13"
OUTPUT_NAME="a09ez5_trace"


for RHIGH in ${RHIGH_LIST[@]}; do
    # 创建日志文件
    LOG_FILE="/home/zzl/Projects/ipole/output/kharma_trace/20250406_output/${OUTPUT_NAME}_Rhigh${RHIGH}_trace.log"
    OUTPUT_DIR="/home/zzl/Projects/ipole/output/kharma_trace/20250406_output/${OUTPUT_NAME}_Rhigh${RHIGH}"
    python job_trace.py --input_dir $FILE_DIR --r_high $RHIGH --file_begin 500 --file_end 1001 --file_step 20 --csv_filename ${OUTPUT_NAME}_Rhigh$RHIGH --h5file_dir $OUTPUT_DIR 2>&1 | tee -a $LOG_FILE

    # 处理以_trace.h5结尾的文件
    for file in $OUTPUT_DIR/*_trace.h5; do
        if [ -f "$file" ]; then
            echo "Processing $file"
            python trace_plot.py --file_name $file
        fi
    done

    python plot_overlay.py --input_dir $OUTPUT_DIR --output_dir $OUTPUT_DIR --spin 0.9375 --eta 0.0
done


