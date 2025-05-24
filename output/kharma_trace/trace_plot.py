import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.colors import LogNorm
import argparse
import multiprocessing as mp
from functools import partial
import os

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10,        # 默认字体大小
    "axes.labelsize": 10,   # 轴标签字体大小
    "legend.fontsize": 10,  # 图例字体大小
    "axes.titlesize": 10,   # 标题字体大小
    "mathtext.fontset": "stix",  # 数学字体设置为 STIX，这个与 Times New Roman 很接近
})

def process_chunk(args, chunk_indices, nstep):
    """处理一个数据块的函数"""
    local_intensity_j = np.zeros((args.resolution_y, args.resolution_x))
    local_intensity_I = np.zeros((args.resolution_y, args.resolution_x))
    
    # 打开HDF5文件
    with h5py.File(args.file_name, 'r') as f:
        for i, j in chunk_indices:
            n = nstep[i,j]
            r_line = f['r'][i,j,:n]
            theta_line = f['th'][i,j,:n]
            j_line = f['j_inv'][i,j,:n,0] * (f['header']['freqcgs'][()]**2)
            I_line = f['stokes'][i,j,:n,0] * (f['header']['freqcgs'][()]**3)
            
            for k in range(len(r_line)-1):
                if r_line[k]*np.sin(theta_line[k]) > args.range_x or r_line[k+1]*np.sin(theta_line[k+1]) > args.range_x or r_line[k]*np.cos(theta_line[k]) > args.range_y/2 or r_line[k+1]*np.cos(theta_line[k+1]) > args.range_y/2:
                    continue
                    
                r_mid = (r_line[k] + r_line[k+1]) / 2
                theta_mid = (theta_line[k] + theta_line[k+1]) / 2
                x_mid = r_mid * np.sin(theta_mid)
                y_mid = r_mid * np.cos(theta_mid)
                
                dI = -I_line[k+1] + I_line[k]
                
                x_idx = np.abs(args.x_grid - x_mid).argmin()
                y_idx = np.abs(args.y_grid - y_mid).argmin()
                
                local_intensity_j[y_idx, x_idx] += j_line[k]
                local_intensity_I[y_idx, x_idx] += dI
    
    return local_intensity_j, local_intensity_I

def split_indices(nstep, n_chunks):
    """将索引分成多个块"""
    indices = [(i, j) for i in range(nstep.shape[0]) for j in range(nstep.shape[1])]
    chunk_size = len(indices) // n_chunks
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trace plot')
    parser.add_argument('--file_name', type=str, required=True, help='Path to the trace file')
    parser.add_argument('--n_processes', type=int, default=mp.cpu_count(), help='Number of processes to use')
    args = parser.parse_args()

    args.resolution_x = 200
    args.resolution_y = 200
    args.range_x = 20
    args.range_y = 20

    # 创建笛卡尔坐标网格
    args.x_grid = np.linspace(0, args.range_x, args.resolution_x)
    args.y_grid = np.linspace(-args.range_y/2, args.range_y/2, args.resolution_y)
    X, Y = np.meshgrid(args.x_grid, args.y_grid)

    # 打开数据文件获取nstep
    with h5py.File(args.file_name, 'r') as f:
        nstep = f['nstep'][:]
        print(nstep.max())
        
        # 准备并行处理
        n_processes = min(args.n_processes, mp.cpu_count())
        chunks = split_indices(nstep, n_processes)
        
        # 创建进程池
        pool = mp.Pool(processes=n_processes)
        
        # 创建部分函数，固定参数
        process_func = partial(process_chunk, args, nstep=nstep)
        
        # 并行处理所有数据块
        results = pool.map(process_func, chunks)
        
        pool.close()
        pool.join()

    # 合并结果
    intensity_j = np.zeros((args.resolution_y, args.resolution_x))
    intensity_I = np.zeros((args.resolution_y, args.resolution_x))
    
    for local_j, local_I in results:
        intensity_j += local_j
        intensity_I += local_I

    # 确保intensity不为负值
    intensity_j = np.maximum(intensity_j, 1e-200)
    intensity_I = np.maximum(intensity_I, 1e-200)

    output_file_name = args.file_name.replace('.h5', '.npz')
    np.savez(output_file_name, intensity_j=intensity_j, intensity_I=intensity_I, 
                              resolution_x=args.resolution_x, resolution_y=args.resolution_y,
                              range_x=args.range_x, range_y=args.range_y)

    # 绘制结果
    fig, axs = plt.subplots(1,2,figsize=(6.8, 4), dpi=300)

    im0 = axs[0].pcolormesh(args.x_grid, args.y_grid, intensity_j, shading='auto', cmap='afmhot'
                            # ,norm=LogNorm(vmin=intensity_j_unpol.max()/1000, vmax=intensity_j_unpol.max())
                    )
    plt.colorbar(im0, ax=axs[0], label='intensity_j', fraction=0.046, pad=0.04)

    im1 = axs[1].pcolormesh(args.x_grid, args.y_grid, intensity_I, shading='auto', cmap='afmhot'
                            # ,norm=LogNorm(vmin=intensity_I.max()/1000, vmax=intensity_I.max())
                            )
    plt.colorbar(im1, ax=axs[1], label='intensity_I', fraction=0.046, pad=0.04)

    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('intensity_j distribution')

    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_title('intensity_I distribution')

    # 添加网格线
    axs[0].grid(True, linestyle='--', alpha=0.5)
    axs[1].grid(True, linestyle='--', alpha=0.5)

    # 优化显示
    axs[0].set_aspect('equal')  # 使x和y轴比例相同
    axs[1].set_aspect('equal')  # 使x和y轴比例相同
    plt.tight_layout()
    plt.savefig(args.file_name.replace('.h5', '.png'), dpi=300)
    plt.show()