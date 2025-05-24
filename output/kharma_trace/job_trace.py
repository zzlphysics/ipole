import subprocess
import pandas as pd
import random
import string
import re
import os
import sys
from scipy.optimize import root_scalar
import argparse
import glob
import h5py


# 生成随机文件名
# def generate_random_filename():
#     return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.h5'

def parse_ipole_output(output):
    Rin_match = re.search(r'Rin: (\d+\.\d+)', output)
    Rin = float(Rin_match.group(1)) if Rin_match else None

    Rout_match = re.search(r'Rout (\d+)', output)
    Rout = float(Rout_match.group(1)) if Rout_match else None

    Xcam_match = re.search(r'Xcam\[\] = ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+)', output)
    Xcam = ','.join(Xcam_match.groups()) if Xcam_match else None

    Dsource_cm_match = re.search(r'Dsource: ([\d.e+-]+) \[cm\]', output)
    Dsource_cm = float(Dsource_cm_match.group(1)) if Dsource_cm_match else None

    Dsource_kpc_match = re.search(r'Dsource: ([\d.e+-]+) \[kpc\]', output)
    Dsource_kpc = float(Dsource_kpc_match.group(1)) if Dsource_kpc_match else None

    FOV_GM_c2_match = re.search(r'FOVx, FOVy: ([\d.e+-]+) ([\d.e+-]+) \[GM/c\^2\]', output)
    FOVx_GM_c2, FOVy_GM_c2 = FOV_GM_c2_match.groups() if FOV_GM_c2_match else (None, None)

    FOV_rad_match = re.search(r'FOVx, FOVy: ([\d.e+-]+) ([\d.e+-]+) \[rad\]', output)
    FOVx_rad, FOVy_rad = FOV_rad_match.groups() if FOV_rad_match else (None, None)

    FOV_muas_match = re.search(r'FOVx, FOVy: ([\d.e+-]+) ([\d.e+-]+) \[muas\]', output)
    FOVx_muas, FOVy_muas = FOV_muas_match.groups() if FOV_muas_match else (None, None)

    Resolution_match = re.search(r'Resolution: (\d+)x(\d+)', output)
    Resolution = ','.join(Resolution_match.groups()) if Resolution_match else None

    scale_match = re.search(r'scale = ([\d.e+-]+)', output)
    scale = float(scale_match.group(1)) if scale_match else None

    imax_match = re.search(r'imax=(\d+)', output)
    imax = int(imax_match.group(1)) if imax_match else None

    jmax_match = re.search(r'jmax=(\d+)', output)
    jmax = int(jmax_match.group(1)) if jmax_match else None

    Imax_match = re.search(r'Imax=([\d.e+-]+)', output)
    Imax = float(Imax_match.group(1)) if Imax_match else None

    Iavg_match = re.search(r'Iavg=([\d.e+-]+)', output)
    Iavg = float(Iavg_match.group(1)) if Iavg_match else None

    freq_match = re.search(r'freq: ([\d.e+-]+)', output)
    freq = float(freq_match.group(1)) if freq_match else None

    Ftot_match = re.search(r'Ftot: ([\d.e+-]+) Jy', output)
    Ftot = float(Ftot_match.group(1)) if Ftot_match else None

    unpol_xfer_match = re.search(r'Ftot: [\d.e+-]+ Jy \(([\d.e+-]+) Jy unpol xfer\)', output)
    unpol_xfer = float(unpol_xfer_match.group(1)) if unpol_xfer_match else None

    nuLnu_match = re.search(r'nuLnu = ([\d.e+-]+) erg/s', output)
    nuLnu = float(nuLnu_match.group(1)) if nuLnu_match else None

    I_match = re.search(r'I,Q,U,V \[Jy\]: ([\d.e+-]+)', output)
    I = float(I_match.group(1)) if I_match else None

    Q_match = re.search(r'I,Q,U,V \[Jy\]: [\d.e+-]+ (-?[\d.e+-]+)', output)
    Q = float(Q_match.group(1)) if Q_match else None

    U_match = re.search(r'I,Q,U,V \[Jy\]: [\d.e+-]+ -?[\d.e+-]+ ([\d.e+-]+)', output)
    U = float(U_match.group(1)) if U_match else None

    V_match = re.search(r'I,Q,U,V \[Jy\]: [\d.e+-]+ -?[\d.e+-]+ [\d.e+-]+ (-?[\d.e+-]+)', output)
    V = float(V_match.group(1)) if V_match else None

    LP_match = re.search(r'LP,CP \[%\]: ([\d.e+-]+)', output)
    LP = float(LP_match.group(1)) if LP_match else None

    CP_match = re.search(r'LP,CP \[%\]: [\d.e+-]+ (-?[\d.e+-]+)', output)
    CP = float(CP_match.group(1)) if CP_match else None
    
    return {
        'Rin': Rin,
        'Rout': Rout,
        'Xcam': Xcam,
        'Dsource_cm': Dsource_cm,
        'Dsource_kpc': Dsource_kpc,
        'FOVx_GM/c^2': float(FOVx_GM_c2) if FOVx_GM_c2 is not None else None,
        'FOVy_GM/c^2': float(FOVy_GM_c2) if FOVy_GM_c2 is not None else None,
        'FOVx_rad': float(FOVx_rad) if FOVx_rad is not None else None,
        'FOVy_rad': float(FOVy_rad) if FOVy_rad is not None else None,
        'FOVx_muas': float(FOVx_muas) if FOVx_muas is not None else None,
        'FOVy_muas': float(FOVy_muas) if FOVy_muas is not None else None,
        'Resolution': Resolution,
        'scale': scale,
        'imax': imax,
        'jmax': jmax,
        'Imax': Imax,
        'Iavg': Iavg,
        'freq': freq,
        'Ftot': Ftot,
        'unpol_xfer': unpol_xfer,
        'nuLnu': nuLnu,
        'I': I,
        'Q': Q,
        'U': U,
        'V': V,
        'LP': LP,
        'CP': CP
    }

def extract_unpol_xfer(output):
    match = re.search(r'Jy \((\d+\.?\d*e?[+-]?\d*) Jy unpol xfer\)', output)
    if match:
        return float(match.group(1))
    else:
        return None

def extract_pol_xfer(output):
    # Ftot: 0.487511 Jy 
    match = re.search(r'Ftot: ([\d.e+-]+) Jy', output)
    if match:
        return float(match.group(1))
    else:
        return None

def run_c(file_path, M_unit, MBH, freq, r_high, thetacam, rotcam, xoff, yoff, output_dir, outfile):
    cmd = [
        './ipole', '-par', 'm87.par',
        '--dump={}'.format(file_path),
        '--thetacam={}'.format(thetacam),
        '--M_unit={}'.format(M_unit),
        '--MBH={}'.format(MBH),
        '--trat_large={}'.format(r_high),
        '--freqcgs={}'.format(freq),
        '--rotcam={}'.format(rotcam),
        '--xoff={}'.format(xoff),
        '--yoff={}'.format(yoff),
        '--outfile={}'.format(output_dir + "/" + outfile),
        '--nx=256',
        '--ny=256',
        '--maxnstep=20000',
        '--trace=1',
        '--trace_stride=1',
        '--trace_outf={}'.format(output_dir + "/" + outfile.replace(".h5", "_trace.h5"))
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr
    print("polarized output: ")
    print(output)
    try:
        parsed_output = parse_ipole_output(output)
    except:
        print("Error parsing output")
        parsed_output = None
    return parsed_output

def run_c_unpol(file_path, M_unit, MBH, r_high, thetacam, rotcam, xoff, yoff):
    cmd = [
        './ipole', '-par', 'm87.par',
        '--dump={}'.format(file_path),
        '--thetacam={}'.format(thetacam),
        '--MBH={}'.format(MBH),
        '--trat_large={}'.format(r_high),
        '--M_unit={}'.format(M_unit),
        '--rotcam={}'.format(rotcam),
        '--xoff={}'.format(xoff),
        '--yoff={}'.format(yoff),
        '--nx=100',
        '--ny=100',
        '--quench_output=1'
        # '--only_unpolarized=1'
    ]
    print("M_unit: ", M_unit)
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr
    print("unpol output: ")
    print(output)
    I_value = extract_pol_xfer(output)
    # print(I_value)
    return I_value



# 创建 ArgumentParser 对象
parser = argparse.ArgumentParser(description='Process run count, CSV file name, and H5 file directory .')

# 添加命令行参数
parser.add_argument('--input_dir', type=str, help='The input directory')
parser.add_argument('--r_high', type=int, help='R_high')
parser.add_argument('--file_begin', type=int, help='file_begin', default=0)
parser.add_argument('--file_end', type=int, help='file_end', default=1000)
parser.add_argument('--file_step', type=int, help='file_step', default=10)
parser.add_argument('--csv_filename', type=str, help='The CSV file name to process')
parser.add_argument('--h5file_dir', type=str, help='The directory for the H5 files')


if __name__ == '__main__':
    args = parser.parse_args()

    # 解析命令行参数
    args = parser.parse_args()

    # 访问参数
    input_dir = args.input_dir
    r_high = args.r_high
    file_begin = args.file_begin
    file_end = args.file_end
    file_step = args.file_step
    csv_filename = args.csv_filename
    h5file_dir = args.h5file_dir

    #判断h5file_dir文件夹是否存在，如果不存在则创建
    if not os.path.exists(h5file_dir):
        os.makedirs(h5file_dir)

    # 定义参数范围
    # thetacam_range = (5,175)
    # MBH_range = (5e9, 8e9)
    # rotcam_range = (0., 360.0)
    # xoff_range = (-32., 32.)
    # yoff_range = (-32., 32.)

    # 初始化CSV文件
    csv_file = csv_filename+'.csv'
    df = pd.DataFrame(columns=['filename', 'thetacam', 'R_high', 'MBH', 'M_unit',
                            'rotcam', 'xoff', 'yoff', 
                            'Rin', 'Rout', 'Xcam', 'Dsource_cm', 'Dsource_kpc', 'FOVx_GM/c^2', 'FOVy_GM/c^2', 
                            'FOVx_rad', 'FOVy_rad', 'FOVx_muas', 'FOVy_muas', 'Resolution', 'scale', 'imax', 'jmax', 
                            'Imax', 'Iavg', 'freq', 'Ftot', 'unpol_xfer', 'nuLnu', 'I', 'Q', 'U', 'V', 'LP', 'CP'])
    # 检查CSV文件是否存在，如果不存在则创建一个空的DataFrame并写入表头
    if not os.path.isfile(csv_file):
        pd.DataFrame(columns=df.columns).to_csv(csv_file, index=False)
    
    # 运行ipole并保存参数到CSV
    pattern = os.path.join(input_dir, "*.out0.*.phdf")
    file_paths = sorted(glob.glob(pattern))  # 对文件路径进行排序
    print("Files number: ", len(file_paths), "begin: ", file_begin, "end: ", file_end, "step: ", file_step)
    if file_begin >= len(file_paths):
        print("file_begin >= len(file_paths), file_begin: ", file_begin, "len(file_paths): ", len(file_paths))
        exit()
    if file_end > len(file_paths):
        print("file_end > len(file_paths), file_end: ", file_end, "len(file_paths): ", len(file_paths))
        exit()
    root_before = None
    root_scal = 0.05
    print([fp for fp in file_paths[file_begin:file_end:file_step]])
    sys.stdout.flush()

    for file_path_remote in file_paths[file_begin:file_end:file_step]:
        file_name = os.path.basename(file_path_remote)

        os.system("cp " + file_path_remote + " " + h5file_dir + "/" + file_name)
        file_path = h5file_dir + "/" + file_name
        
        try:
            f = h5py.File(file_path, "r")
        except:
            raise ValueError("File not found")

        # Read in the timestep attributes
        info = f["Info"]
        try:
            time = info.attrs["Time"]
        except:
            raise ValueError("Time not found in Info")

        
        thetacam = 163 #random.uniform(*thetacam_range)
        MBH = 6.5e9 #random.uniform(*MBH_range)
        rotcam = 180 #random.uniform(*rotcam_range)
        xoff = 0 #random.uniform(*xoff_range)
        yoff = 0 #random.uniform(*yoff_range)


        print('Running ipole with parameters:')
        print('file_name:', file_name)
        print('thetacam:', thetacam)
        # print('MBH:', MBH)
        print('rotcam:', rotcam)
        print('xoff:', xoff)
        print('yoff:', yoff)


        nll = lambda args:run_c_unpol(file_path, args, MBH, r_high, thetacam, rotcam, xoff, yoff)-0.505
        if root_before is not None:
            bracket = [(1-root_scal)*root_before, (1+root_scal)*root_before]
        else:
            bracket = [1e20, 1e30]

        continue_try = True
        run_count = 0
        while continue_try:
            run_count += 1
            continue_try = False
            try:
                soln = root_scalar(nll, bracket=bracket, method='brentq', xtol = 2e-2,rtol = 2e-2, maxiter=30)
            # soln = root_scalar(nll, x0=1e5, x1=1e6, method='secant', xtol = 0.1,rtol = 0.02, maxiter=20)
            except Exception as e:
                print("find root error:", e)
                continue_try = True
                bracket = [0.1 *bracket[0], 10*bracket[1]]
                if root_before is not None:
                    root_scal +=0.1
            
            if run_count > 5:
                continue_try = False
                print("run_count > 5, continue_try = False")

        if soln.converged == False:
            print("soln.converged = False!, file_name:", file_path)
            continue
        root_before = soln.root

        for freq in [ 230e9]:
            outfile = file_name.replace(".phdf", f"_{int(freq/1e9)}GHz.h5")
            print('outfile:', outfile)
            parsed_output = run_c(file_path, soln.root, MBH, freq, r_high, thetacam, rotcam, xoff, yoff, h5file_dir, outfile)
            if parsed_output is None:
                continue
            print("parsed_output:")
            print(parsed_output)
            print("---------------------------------------------------")
            print("\n")

            # 保存参数到CSV
            new_data = {
                'filename': outfile,
                'thetacam': thetacam,
                'R_high': r_high,
                'MBH': MBH,
                'M_unit': soln.root, 
                'rotcam': rotcam,
                'xoff': xoff,
                'yoff': yoff,
                **parsed_output
            }

            # 使用追加模式写入新数据
            df = pd.DataFrame([new_data])
            df.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)

        os.system("rm " + file_path)
        # 强制刷新标准输出缓冲区
        sys.stdout.flush()

    print("任务完成，所有数据已生成并保存到CSV文件中。")



