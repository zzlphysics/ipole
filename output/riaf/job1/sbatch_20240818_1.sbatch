#!/bin/bash

#SBATCH --job-name=ipole
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --time=30-00:00:00
#SBATCH --partition=72c512g
#SBATCH --output=res_ipole_%j.out
#SBATCH --error=res_ipole_%j.err

spack env activate ipole

# 定义任务数量
num_tasks=8

# 并行运行任务
for i in $(seq 1 $num_tasks); do
    srun --exclusive -n1 --output=res_ipole_%j_task${i}.out --error=res_ipole_%j_task${i}_%j.err \
        python /home/zhangzelin/projects/ipole/output/riaf/job1/job_20240818.py \
        --run_count=4000 --csv_filename=task${i} --h5file_dir=files &
done

wait