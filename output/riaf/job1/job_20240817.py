import subprocess
import pandas as pd
import random
import string
import re
import os
import sys

# 定义参数范围
a_range = (-0.99, 0.99)  # 范围 (min, max)
Ne_unit_range = (1e4, 1e8)
Te_unit_range = (1e7, 1e12)
disk_h_range = (0.1, 0.6)
MBH_range = (5e9, 8e9)
pow_nth_range = (-1.2, -1.0)
pow_T_range = (-0.6, -1.0)
keplerian_factor_range = (0, 1.0)
fluid_dirction_range = [1.0, -1.0]

# 初始化CSV文件
csv_file = 'output.csv'
df = pd.DataFrame(columns=['filename', 'a', 'Ne_unit', 'Te_unit', 'disk_h', 'MBH', 'pow_nth', 'pow_T', 'keplerian_factor', 'fluid_dirction', 
                           'Rin', 'Rout', 'Xcam', 'Dsource_cm', 'Dsource_kpc', 'FOVx_GM/c^2', 'FOVy_GM/c^2', 
                           'FOVx_rad', 'FOVy_rad', 'FOVx_muas', 'FOVy_muas', 'Resolution', 'scale', 'imax', 'jmax', 
                           'Imax', 'Iavg', 'freq', 'Ftot', 'unpol_xfer', 'nuLnu', 'I', 'Q', 'U', 'V', 'LP', 'CP'])

# 生成随机文件名
def generate_random_filename():
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=10)) + '.h5'

# 解析ipole输出
def parse_ipole_output(output):
    Rin_match = re.search(r'Rin: (\d+\.\d+)', output)
    Rin = float(Rin_match.group(1)) if Rin_match else None

    Rout_match = re.search(r'Rout (\d+)', output)
    Rout = float(Rout_match.group(1)) if Rout_match else None

    Xcam_match = re.search(r'Xcam\[\] = ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+) ([\d.e+-]+)', output)
    Xcam = ','.join(Xcam_match.groups()) if Xcam_match else None

    Dsource_cm_match = re.search(r'Dsource: ([\d.e+]+) \[cm\]', output)
    Dsource_cm = float(Dsource_cm_match.group(1)) if Dsource_cm_match else None

    Dsource_kpc_match = re.search(r'Dsource: ([\d.e+]+) \[kpc\]', output)
    Dsource_kpc = float(Dsource_kpc_match.group(1)) if Dsource_kpc_match else None

    FOV_GM_c2_match = re.search(r'FOVx, FOVy: ([\d.]+) ([\d.]+) \[GM/c\^2\]', output)
    FOVx_GM_c2, FOVy_GM_c2 = FOV_GM_c2_match.groups() if FOV_GM_c2_match else (None, None)

    FOV_rad_match = re.search(r'FOVx, FOVy: ([\d.e-]+) ([\d.e-]+) \[rad\]', output)
    FOVx_rad, FOVy_rad = FOV_rad_match.groups() if FOV_rad_match else (None, None)

    FOV_muas_match = re.search(r'FOVx, FOVy: ([\d.]+) ([\d.]+) \[muas\]', output)
    FOVx_muas, FOVy_muas = FOV_muas_match.groups() if FOV_muas_match else (None, None)

    Resolution_match = re.search(r'Resolution: (\d+)x(\d+)', output)
    Resolution = ','.join(Resolution_match.groups()) if Resolution_match else None

    scale_match = re.search(r'scale = ([\d.e+]+)', output)
    scale = float(scale_match.group(1)) if scale_match else None

    imax_match = re.search(r'imax=(\d+)', output)
    imax = int(imax_match.group(1)) if imax_match else None

    jmax_match = re.search(r'jmax=(\d+)', output)
    jmax = int(jmax_match.group(1)) if jmax_match else None

    Imax_match = re.search(r'Imax=([\d.e-]+)', output)
    Imax = float(Imax_match.group(1)) if Imax_match else None

    Iavg_match = re.search(r'Iavg=([\d.e-]+)', output)
    Iavg = float(Iavg_match.group(1)) if Iavg_match else None

    freq_match = re.search(r'freq: ([\d.e+]+)', output)
    freq = float(freq_match.group(1)) if freq_match else None

    Ftot_match = re.search(r'Ftot: ([\d.]+) Jy', output)
    Ftot = float(Ftot_match.group(1)) if Ftot_match else None

    unpol_xfer_match = re.search(r'Ftot: [\d.]+ Jy \(([\d.]+) Jy unpol xfer\)', output)
    unpol_xfer = float(unpol_xfer_match.group(1)) if unpol_xfer_match else None

    nuLnu_match = re.search(r'nuLnu = ([\d.e+]+) erg/s', output)
    nuLnu = float(nuLnu_match.group(1)) if nuLnu_match else None

    I_match = re.search(r'I,Q,U,V \[Jy\]: ([\d.e-]+)', output)
    I = float(I_match.group(1)) if I_match else None

    Q_match = re.search(r'I,Q,U,V \[Jy\]: [\d.e-]+ (-?[\d.e-]+)', output)
    Q = float(Q_match.group(1)) if Q_match else None

    U_match = re.search(r'I,Q,U,V \[Jy\]: [\d.e-]+ -?[\d.e-]+ ([\d.e-]+)', output)
    U = float(U_match.group(1)) if U_match else None

    V_match = re.search(r'I,Q,U,V \[Jy\]: [\d.e-]+ -?[\d.e-]+ [\d.e-]+ (-?[\d.e-]+)', output)
    V = float(V_match.group(1)) if V_match else None

    LP_match = re.search(r'LP,CP \[%\]: ([\d.e-]+)', output)
    LP = float(LP_match.group(1)) if LP_match else None

    CP_match = re.search(r'LP,CP \[%\]: [\d.e-]+ (-?[\d.e-]+)', output)
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

# 运行ipole并保存参数到CSV
for _ in range(5000):  # 生成10个数据集
    a = random.uniform(*a_range)
    Ne_unit = random.uniform(*Ne_unit_range)
    Te_unit = random.uniform(*Te_unit_range)
    disk_h = random.uniform(*disk_h_range)
    MBH = random.uniform(*MBH_range)
    pow_nth = random.uniform(*pow_nth_range)
    pow_T = random.uniform(*pow_T_range)
    keplerian_factor = random.uniform(*keplerian_factor_range)
    infall_factor = 1.0 - keplerian_factor
    fluid_dirction = random.choice(fluid_dirction_range)
    outfile = generate_random_filename()

    print('Running ipole with parameters:')
    print('a:', a)
    print('Ne_unit:', Ne_unit)
    print('Te_unit:', Te_unit)
    print('disk_h:', disk_h)
    print('MBH:', MBH)
    print('pow_nth:', pow_nth)
    print('pow_T:', pow_T)
    print('keplerian_factor:', keplerian_factor)
    print('infall_factor:', infall_factor)
    print('fluid_dirction:', fluid_dirction)
    print('outfile:', outfile)


    cmd = [
        './ipole', '-par', 'm87.par',
        '--a={}'.format(a),
        '--Ne_unit={}'.format(Ne_unit),
        '--Te_unit={}'.format(Te_unit),
        '--disk_h={}'.format(disk_h),
        '--MBH={}'.format(MBH),
        '--pow_nth={}'.format(pow_nth),
        '--pow_T={}'.format(pow_T),
        '--keplerian_factor={}'.format(keplerian_factor),
        '--infall_factor={}'.format(infall_factor),
        '--outfile={}'.format("files/"+outfile)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    # print(result)
    output = result.stderr
    print(output)

    parsed_output = parse_ipole_output(output)
    print("parsed_output:")
    print(parsed_output)
    print("---------------------------------------------------")
    print("\n")

    # 保存参数到CSV
    new_data = {
        'filename': outfile,
        'a': a,
        'Ne_unit': Ne_unit,
        'Te_unit': Te_unit,
        'disk_h': disk_h,
        'MBH': MBH,
        'pow_nth': pow_nth,
        'pow_T': pow_T,
        'keplerian_factor': keplerian_factor,
        'fluid_dirction' : fluid_dirction,
        **parsed_output
    }


    # 检查CSV文件是否存在，如果不存在则创建一个空的DataFrame并写入表头
    if not os.path.isfile(csv_file):
        pd.DataFrame(columns=df.columns).to_csv(csv_file, index=False)

    # 使用追加模式写入新数据
    df = pd.DataFrame([new_data])
    df.to_csv(csv_file, mode='a', header=not os.path.isfile(csv_file), index=False)

    # 强制刷新标准输出缓冲区
    sys.stdout.flush()

print("任务完成，所有数据已生成并保存到CSV文件中。")