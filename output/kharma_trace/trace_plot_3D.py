import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import h5py
from matplotlib.colors import LogNorm
import argparse
import multiprocessing as mp
from functools import partial
import os
import pyvista as pv
from pyvista import themes

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 12,
    "axes.titlesize": 12,
    "mathtext.fontset": "stix",
})

def process_chunk(args, chunk_indices, nstep):
    """处理一个数据块的函数"""
    local_intensity_j = np.zeros((args.resolution_z, args.resolution_y, args.resolution_x))
    local_intensity_I = np.zeros((args.resolution_z, args.resolution_y, args.resolution_x))
    
    with h5py.File(args.file_name, 'r') as f:
        for i, j in chunk_indices:
            n = nstep[i,j]
            r_line = f['r'][i,j,:n]
            theta_line = f['th'][i,j,:n]
            phi_line = f['phi'][i,j,:n]
            j_line = f['j_inv'][i,j,:n,0] * (f['header']['freqcgs'][()]**2)
            I_line = f['stokes'][i,j,:n,0] * (f['header']['freqcgs'][()]**3)
            
            for k in range(len(r_line)-1):
                r_mid = (r_line[k] + r_line[k+1]) / 2
                theta_mid = (theta_line[k] + theta_line[k+1]) / 2
                phi_mid = (phi_line[k] + phi_line[k+1]) / 2
                
                # 球坐标转笛卡尔坐标
                x_mid = r_mid * np.sin(theta_mid) * np.cos(phi_mid)
                y_mid = r_mid * np.sin(theta_mid) * np.sin(phi_mid)
                z_mid = r_mid * np.cos(theta_mid)
                
                # 检查是否在计算域内
                if abs(x_mid) > args.range_x/2 or abs(y_mid) > args.range_y/2 or abs(z_mid) > args.range_z/2:
                    continue
                
                dI = -I_line[k+1] + I_line[k]
                
                # 找到最近的网格点
                x_idx = np.abs(args.x_grid - x_mid).argmin()
                y_idx = np.abs(args.y_grid - y_mid).argmin()
                z_idx = np.abs(args.z_grid - z_mid).argmin()
                
                local_intensity_j[z_idx, y_idx, x_idx] += j_line[k]
                local_intensity_I[z_idx, y_idx, x_idx] += dI
    
    return local_intensity_j, local_intensity_I

def split_indices(nstep, n_chunks):
    """将索引分成多个块"""
    indices = [(i, j) for i in range(nstep.shape[0]) for j in range(nstep.shape[1])]
    chunk_size = len(indices) // n_chunks
    return [indices[i:i + chunk_size] for i in range(0, len(indices), chunk_size)]

def create_volume_visualization(intensity, x_grid, y_grid, z_grid, filename, title, colormap='plasma', opacity=None):
    """创建体积渲染的可视化"""
    # 创建网格
    grid = pv.UniformGrid()
    grid.dimensions = np.array(intensity.shape) + 1
    grid.origin = (x_grid[0], y_grid[0], z_grid[0])
    grid.spacing = (
        (x_grid[-1] - x_grid[0]) / len(x_grid),
        (y_grid[-1] - y_grid[0]) / len(y_grid),
        (z_grid[-1] - z_grid[0]) / len(z_grid)
    )
    
    # 将数据附加到网格
    grid.cell_data["values"] = intensity.flatten(order="F")
    
    # 设置pyvista主题为暗色
    theme = themes.DarkTheme()
    pv.set_plot_theme(theme)
    
    # 创建可视化
    p = pv.Plotter(off_screen=True, window_size=(1200, 800))
    p.background_color = 'black'
    
    # 添加中心黑洞
    sphere = pv.Sphere(radius=2.5, center=(0, 0, 0))
    p.add_mesh(sphere, color='gray', opacity=0.7)
    
    # 计算用于体积渲染的不透明度
    if opacity is None:
        # 找到合适的阈值
        min_val = np.percentile(intensity[intensity > 1e-190], 50)
        max_val = np.percentile(intensity, 99.5)
        
        # 把阈值缩放到[0,1]范围
        rng = max_val - min_val
        opacity = np.clip((intensity - min_val) / rng, 0, 1)
        opacity = opacity ** 2  # 平方使低值更透明
        opacity = opacity * 0.95  # 降低整体不透明度
    
    # 添加体积渲染
    p.add_volume(grid, cmap=colormap, opacity=opacity, shade=True)
    
    # 添加坐标轴
    p.add_axes(labels_off=False)
    
    # 添加标尺
    p.add_ruler([-10, -10, -10], [-5, -10, -10], title='5 GM/c²', color='white')
    
    # 添加标题
    p.add_title(title, font_size=18, color='white')
    
    # 设置相机位置
    p.camera_position = [(30, 20, 15), (0, 0, 0), (0, 0, 1)]
    p.camera.zoom(1.2)
    
    # 保存图像
    p.show(screenshot=filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Trace plot')
    parser.add_argument('--file_name', type=str, required=True, help='Path to the trace file')
    parser.add_argument('--n_processes', type=int, default=mp.cpu_count(), help='Number of processes to use')
    args = parser.parse_args()

    # 设置分辨率和范围
    args.resolution_x = 50
    args.resolution_y = 50
    args.resolution_z = 50
    args.range_x = 20
    args.range_y = 20
    args.range_z = 20

    # 创建3D网格
    args.x_grid = np.linspace(-args.range_x/2, args.range_x/2, args.resolution_x)
    args.y_grid = np.linspace(-args.range_y/2, args.range_y/2, args.resolution_y)
    args.z_grid = np.linspace(-args.range_z/2, args.range_z/2, args.resolution_z)

    # 检查npz文件是否存在
    output_file_name = args.file_name.replace('.h5', '_3D.npz')
    if os.path.exists(output_file_name):
        print(f"找到已存在的数据文件: {output_file_name}，直接加载")
        data = np.load(output_file_name)
        intensity_j = data['intensity_j']
        intensity_I = data['intensity_I']
        args.x_grid = data['x_grid']
        args.y_grid = data['y_grid']
        args.z_grid = data['z_grid']
    else:
        print(f"未找到数据文件，开始计算...")
        # 数据处理部分
        with h5py.File(args.file_name, 'r') as f:
            nstep = f['nstep'][:]
            print(f"Maximum steps: {nstep.max()}")
            
            n_processes = min(args.n_processes, mp.cpu_count())
            chunks = split_indices(nstep, n_processes)
            
            pool = mp.Pool(processes=n_processes)
            process_func = partial(process_chunk, args, nstep=nstep)
            results = pool.map(process_func, chunks)
            
            pool.close()
            pool.join()

        # 合并结果
        intensity_j = np.zeros((args.resolution_z, args.resolution_y, args.resolution_x))
        intensity_I = np.zeros((args.resolution_z, args.resolution_y, args.resolution_x))
        
        for local_j, local_I in results:
            intensity_j += local_j
            intensity_I += local_I

        # 确保intensity不为负值
        intensity_j = np.maximum(intensity_j, 1e-200)
        intensity_I = np.maximum(intensity_I, 1e-200)

        # 保存数据
        np.savez(output_file_name, 
                intensity_j=intensity_j, 
                intensity_I=intensity_I,
                x_grid=args.x_grid,
                y_grid=args.y_grid,
                z_grid=args.z_grid)
        print(f"数据已保存至: {output_file_name}")

    # 使用pyvista进行体积渲染
    print("正在创建体积渲染...")
    
    # 渲染发射率j
    create_volume_visualization(
        intensity_j, 
        args.x_grid, args.y_grid, args.z_grid,
        args.file_name.replace('.h5', '_3D_j.png'),
        "3D Distribution of Emission",
        colormap='inferno'
    )
    
    # 渲染辐射增量I的绝对值
    create_volume_visualization(
        np.abs(intensity_I), 
        args.x_grid, args.y_grid, args.z_grid,
        args.file_name.replace('.h5', '_3D_I.png'),
        "3D Distribution of Radiation Increment",
        colormap='inferno'
    )
    
    print(f"渲染完成，图像已保存至: {args.file_name.replace('.h5', '_3D_j.png')} 和 {args.file_name.replace('.h5', '_3D_I.png')}")