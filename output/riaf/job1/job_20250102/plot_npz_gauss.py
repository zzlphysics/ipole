import os
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter
# 对于所有输入的npz文件,读取generated_image, original_label,保存到新的npz文件中,
# 新的npz文件名是原文件名+_modified.npz
# 输入文件由命令行给定

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理npz格式文件')
    parser.add_argument('--input_files', type=str, nargs='+', required=True, help='输入npz文件列表，支持通配符')
    args = parser.parse_args()

    # 使用glob处理通配符
    import glob
    all_files = []
    for pattern in args.input_files:
        all_files.extend(glob.glob(pattern))

    if not all_files:
        print("错误: 没有找到匹配的npz文件")
        exit(1)

    for input_file in all_files:
        if not input_file.endswith('.npz'):
            print(f"警告: 跳过非npz文件 {input_file}")
            continue
            
        try:
            # 读取输入文件
            data = np.load(input_file)
            I_rot = data['I_rot']
            normalized_parm_array = data['normalized_parm_array']
            sigma = np.sqrt(1/12)
            mean_mbh = 6.5e9
            std_mbh = 3e9*sigma
            mbh = mean_mbh + std_mbh * normalized_parm_array[3]

            G_CONS = 6.6742e-8
            C_CONS = 2.99792458e10
            M_SUN = 1.989e33
            PC = 3.085678e18
            lunit = G_CONS * mbh * M_SUN / (C_CONS * C_CONS)
            MUAS_PER_RAD = 2.06265e11
            fovx_dsource = 160
            dsource_pc = 16.8e6
            dsource = dsource_pc * PC
            fov_to_d = dsource / lunit / MUAS_PER_RAD
            dx = fovx_dsource * fov_to_d

            nx = I_rot.shape[0]
            rad2muas = np.pi/180/3600/1000000
            sigma_10 = 10 / (2 * np.sqrt(2 * np.log(2))) / ((dx*lunit/nx)/(dsource)/rad2muas)
            sigma_20 = 20 / (2 * np.sqrt(2 * np.log(2))) / ((dx*lunit/nx)/(dsource)/rad2muas)

            I_rot_gauss10 = gaussian_filter(I_rot, sigma_10)
            I_rot_gauss20 = gaussian_filter(I_rot, sigma_20)

            # 保存到新的npz文件中
            output_file = input_file.replace('.npz', '_convolved.npz')
            np.savez(output_file, I_rot=I_rot, I_rot_gauss10=I_rot_gauss10, I_rot_gauss20=I_rot_gauss20, normalized_parm_array=normalized_parm_array)
            print(f"成功处理文件: {input_file} -> {output_file}")
            
        except Exception as e:
            print(f"处理文件 {input_file} 时出错: {str(e)}")
            continue
