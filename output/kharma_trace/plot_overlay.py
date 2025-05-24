import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
import os
import glob
import sys
import argparse

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,        # 默认字体大小
    "axes.labelsize": 10,   # 轴标签字体大小
    "legend.fontsize": 10,  # 图例字体大小
    "axes.titlesize": 10,   # 标题字体大小
    "mathtext.fontset": "stix",  # 数学字体设置为 STIX，这个与 Times New Roman 很接近
})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot overlay of intensity_j and intensity_I')
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing npz files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for the plot')
    parser.add_argument('--spin', type=float, required=True, help='Spin of the trace')
    parser.add_argument('--eta', type=float, required=True, help='Eta of the trace')
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    a = args.spin
    eta = args.eta

    discriminant = -4*a**4 + 4*a**6 - 36*a**2*eta + eta*(32 + 27*eta)
    
    if discriminant >= 0:
        term1 = 8 - 9*a**2 + (27*eta)/2 + (3*np.sqrt(3)*np.sqrt(discriminant))/2
        term2 = 16 - 18*a**2 + 27*eta + 3*np.sqrt(3)*np.sqrt(discriminant)
        rh = (4 - (2*(-4 + 3*a**2))/np.power(term1, 1/3) + 
              np.power(2, 2/3)*np.power(term2, 1/3))/6
    else:
        term1 = 4/3 - a**2
        term2 = (3*np.sqrt(3)*(16/27 - (2*a**2)/3 + eta))/(2*np.power(term1, 1.5))
        rh = 2/3 + (2*np.sqrt(term1)*np.cos(np.arccos(term2)/3))/np.sqrt(3)
    print(f'rh = {rh}')
    
    # 创建一个列表来存储所有intensity_j
    intensity_j_list = []
    intensity_I_list = []
    range_x_plot = 0
    range_y_plot = 0

    input_files = glob.glob(f'{input_dir}/*_trace.npz')
    print(f'Files number: {len(input_files)}')
    # 循环处理每个文件
    for file_path in input_files:
        try:
            # 这里可以添加对每个文件的具体处理逻辑
            # print(f"正在处理文件: {file_path}")
            
            # 示例：加载npz文件
            data = np.load(file_path)
            
            intensity_j = data['intensity_j']
            intensity_I = data['intensity_I']
            resolution_x = data['resolution_x']
            resolution_y = data['resolution_y']
            range_x = data['range_x']
            range_y = data['range_y']

            intensity_j_list.append(intensity_j)
            intensity_I_list.append(intensity_I)
            range_x_plot = range_x
            range_y_plot = range_y
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")
            continue

    intensity_j_list = np.array(intensity_j_list)
    intensity_I_list = np.array(intensity_I_list)

    # 计算所有intensity_j的平均值，并翻转
    intensity_j_mean = np.flip(np.mean(intensity_j_list, axis=0), axis=0)
    intensity_I_mean = np.flip(np.mean(intensity_I_list, axis=0), axis=0)

    # 绘制intensity_j的平均值
    fig0, ax0 = plt.subplots(figsize=(4.2, 3.6), dpi=300)
    im0 = ax0.imshow(intensity_j_mean, cmap='afmhot', extent=[0, range_x_plot, -range_y_plot/2, range_y_plot/2])
    # 设置colorbar的大小和刻度
    cbar0 = plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    # 设置colorbar的刻度
    vmin, vmax = im0.get_clim()
    cbar0.set_ticks(np.linspace(vmin, vmax, 5))
    # 添加半径为rh的白色圆形
    circle = plt.Circle((0, 0), rh, fill=False, color='white', linewidth=1)
    ax0.add_patch(circle)
    ax0.set_title(f'a = {a}, $\eta$ = {eta}')
    ax0.set_xlabel(r'$r ~\cos \theta$ [M]')
    ax0.set_ylabel(r'$r ~\sin \theta$ [M]')
    ax0.set_aspect('equal')
    ax0.set_xlim(0, 16)
    ax0.set_ylim(-8, 8)
    # 设置x轴和y轴的刻度
    x_ticks = np.linspace(0, 16, 5)
    y_ticks = np.linspace(-8, 8, 5)
    
    ax0.set_xticks(x_ticks)
    ax0.set_yticks(y_ticks)
    ax0.set_xticklabels([f'{x:.1f}' for x in x_ticks])
    ax0.set_yticklabels([f'{y:.1f}' for y in y_ticks])
    plt.savefig(f'{output_dir}/intensity_j_mean.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    fig1, ax1 = plt.subplots(figsize=(4.2, 3.6), dpi=300)
    im1 = ax1.imshow(intensity_I_mean, cmap='afmhot', extent=[0, range_x_plot, -range_y_plot/2, range_y_plot/2])
    circle = plt.Circle((0, 0), rh, fill=False, color='white', linewidth=1)
    ax1.add_patch(circle)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    vmin, vmax = im1.get_clim()
    cbar1.set_ticks(np.linspace(vmin, vmax, 5))
    ax1.set_title(f'a = {a}, $\eta$ = {eta}')
    ax1.set_xlabel(r'$r ~\cos \theta$ [M]')
    ax1.set_ylabel(r'$r ~\sin \theta$ [M]')
    ax1.set_aspect('equal')
    ax1.set_xlim(0, 16)
    ax1.set_ylim(-8, 8)
    x_ticks = np.linspace(0, 16, 5)
    y_ticks = np.linspace(-8, 8, 5)
    ax1.set_xticks(x_ticks)
    ax1.set_yticks(y_ticks)
    ax1.set_xticklabels([f'{x:.1f}' for x in x_ticks])
    ax1.set_yticklabels([f'{y:.1f}' for y in y_ticks])
    plt.savefig(f'{output_dir}/intensity_I_mean.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    