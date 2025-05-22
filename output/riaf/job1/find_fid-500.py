# 找到 /root/Projects/ipole/output/riaf/job1/output20250102.csv 中最后500个文件，第一列为文件名，文件目录在 /root/hdd256/ipole/files_20250102 ，然后将这些文件复制到 /root/hdd256/ipole/files_20250102_fid-500 目录下
import pandas as pd
import os
import shutil
# 读取 CSV 文件
df = pd.read_csv('/root/Projects/ipole/output/riaf/job1/output20250102.csv')

# 获取最后500个文件
last_files = df.iloc[2200:2400]
print(len(last_files))
# 创建目标目录
target_dir = '/root/hdd256/ipole/files_20250102_ResNet-200'
os.makedirs(target_dir, exist_ok=True)

# 复制文件
for index, row in last_files.iterrows():
    file_name = row['filename']
    src_path = os.path.join('/root/hdd256/ipole/files_20250102', file_name)
    dst_path = os.path.join(target_dir, file_name)
    shutil.copy(src_path, dst_path)

print(f"已复制 {len(last_files)} 个文件到 {target_dir}")
