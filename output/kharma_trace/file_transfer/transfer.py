#!/usr/bin/env python3
import os
import hashlib
import logging
import json
import time
from pathlib import Path
import paramiko
from scp import SCPClient
from tqdm import tqdm

class FileTransfer:
    def __init__(self, source_dir, target_dir, hostname, username, port=22, password=None, key_filename=None, 
                 log_level=logging.INFO, verify_existing=True):
        self.source_dir = Path(source_dir)  # 远程源目录
        self.target_dir = Path(target_dir)  # 本地目标目录
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.key_filename = key_filename
        self.verify_existing = verify_existing  # 是否验证已存在的文件
        
        # 传输统计
        self.start_time = None
        self.transferred_size = 0
        self.pbar = None
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # 生成带时间戳的日志文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_filename = f'file_transfer_{timestamp}.log'
        
        # 文件处理器 - 记录所有日志
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # 控制台处理器 - 只记录错误和警告
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 设置paramiko日志级别
        logging.getLogger("paramiko").setLevel(logging.WARNING)
        
        # 进度文件
        self.progress_file = 'transfer_progress.json'
        self.progress = self._load_progress()
        
        # 确保本地目标目录存在
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计远程文件信息
        self.total_files = 0
        self.total_size = 0
        self._count_remote_files()
        
    def _load_progress(self):
        """加载传输进度"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                return json.load(f)
        return {'completed_files': [], 'failed_files': []}
    
    def _save_progress(self):
        """保存传输进度"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)
    
    def _calculate_sha256(self, file_path):
        """计算文件的SHA256哈希值"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _count_remote_files(self):
        """统计远程目录中的文件数量和总大小"""
        try:
            ssh = self._create_ssh_client(show_connection_info=True)
            self.logger.info(f"开始统计远程目录: {self.source_dir}")
            
            # 使用find命令统计文件
            stdin, stdout, stderr = ssh.exec_command(f'find {self.source_dir} -type f')
            files = stdout.read().decode().splitlines()
            self.total_files = len(files)
            
            # 使用du命令统计总大小
            stdin, stdout, stderr = ssh.exec_command(f'du -sb {self.source_dir}')
            output = stdout.read().decode().strip()
            error = stderr.read().decode().strip()
            
            if error:
                self.logger.error(f"du命令执行出错: {error}")
                raise Exception(f"du命令执行出错: {error}")
                
            if not output:
                self.logger.error("du命令没有输出")
                raise Exception("du命令没有输出")
                
            try:
                self.total_size = int(output.split()[0])
            except (IndexError, ValueError) as e:
                self.logger.error(f"解析du命令输出失败: {output}")
                raise Exception(f"解析du命令输出失败: {output}")
            
            ssh.close()
            self.logger.info(f"远程目录统计完成: 共 {self.total_files} 个文件，总大小 {self.total_size / 1024 / 1024:.2f} MB")
            print(f"远程目录统计完成: 共 {self.total_files} 个文件，总大小 {self.total_size / 1024 / 1024:.2f} MB")
        except Exception as e:
            self.logger.error(f"统计远程目录失败: {str(e)}")
            raise
    
    def _create_ssh_client(self, show_connection_info=False):
        """创建SSH客户端"""
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        if show_connection_info:
            # 只在初始连接时显示连接信息
            if self.key_filename:
                ssh.connect(self.hostname, port=self.port, username=self.username, key_filename=self.key_filename)
            else:
                ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password)
        else:
            # 后续连接不显示连接信息
            if self.key_filename:
                ssh.connect(self.hostname, port=self.port, username=self.username, key_filename=self.key_filename, banner_timeout=200)
            else:
                ssh.connect(self.hostname, port=self.port, username=self.username, password=self.password, banner_timeout=200)
        return ssh
    
    def _verify_file(self, local_path, remote_path):
        """验证本地和远程文件的完整性"""
        try:
            # 获取本地文件哈希
            local_hash = self._calculate_sha256(local_path)
            
            # 创建SSH客户端
            ssh = self._create_ssh_client()
            
            # 获取远程文件哈希
            stdin, stdout, stderr = ssh.exec_command(f'sha256sum {remote_path}')
            remote_hash = stdout.read().decode().split()[0]
            
            ssh.close()
            
            return local_hash == remote_hash
        except Exception as e:
            self.logger.error(f"验证文件失败: {str(e)}")
            return False
    
    def _format_size(self, size_bytes):
        """格式化文件大小显示"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"
    
    def _update_progress(self, file_size):
        """更新传输进度"""
        if self.pbar is None:
            self.pbar = tqdm(
                total=self.total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
                desc="传输进度",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )
        self.pbar.update(file_size)
    
    def transfer_file(self, remote_path, local_path):
        """从远程传输单个文件到本地"""
        try:
            # 获取文件大小
            ssh = self._create_ssh_client(show_connection_info=False)
            stdin, stdout, stderr = ssh.exec_command(f'stat -c %s {remote_path}')
            file_size = int(stdout.read().decode().strip())
            ssh.close()
            
            # 检查文件是否已经传输过
            if str(remote_path) in self.progress['completed_files']:
                if self.verify_existing:
                    # 验证已存在文件的完整性
                    self.logger.info(f"正在验证文件: {remote_path}")
                    if self._verify_file(local_path, remote_path):
                        self.logger.info(f"文件已验证，跳过: {remote_path}")
                        # 更新进度条
                        self._update_progress(file_size)
                        return True
                    else:
                        self.logger.warning(f"文件验证失败，重新传输: {remote_path}")
                        print(f"文件验证失败，重新传输: {remote_path}")
                else:
                    self.logger.info(f"文件已存在，跳过: {remote_path}")
                    print(f"文件已存在，跳过: {remote_path}")
                    # 更新进度条
                    self._update_progress(file_size)
                    return True
            
            self.logger.info(f"开始传输文件: {remote_path}")
            
            # 创建SSH客户端（不显示连接信息）
            ssh = self._create_ssh_client(show_connection_info=False)
            
            # 创建SCP客户端
            scp = SCPClient(ssh.get_transport())
            
            # 确保本地目录存在
            local_dir = os.path.dirname(local_path)
            os.makedirs(local_dir, exist_ok=True)
            
            # 传输文件
            scp.get(remote_path, local_path)
            
            # 更新进度
            self._update_progress(file_size)
            
            # 验证文件
            if self._verify_file(local_path, remote_path):
                self.logger.info(f"文件传输成功: {remote_path}")
                self.progress['completed_files'].append(str(remote_path))
                self._save_progress()
                return True
            else:
                self.logger.error(f"文件验证失败: {remote_path}")
                self.progress['failed_files'].append(str(remote_path))
                self._save_progress()
                return False
                
        except Exception as e:
            self.logger.error(f"传输文件失败: {str(e)}")
            self.progress['failed_files'].append(str(remote_path))
            self._save_progress()
            return False
        finally:
            try:
                scp.close()
                ssh.close()
            except:
                pass
    
    def transfer_directory(self):
        """从远程传输整个目录到本地"""
        try:
            # 初始化开始时间
            self.start_time = time.time()
            
            # 创建SSH客户端（显示初始连接信息）
            ssh = self._create_ssh_client(show_connection_info=True)
            
            # 检查远程目录是否存在
            stdin, stdout, stderr = ssh.exec_command(f'test -d {self.source_dir} && echo "exists"')
            if not stdout.read().decode().strip():
                raise FileNotFoundError(f"远程目录不存在: {self.source_dir}")
            
            # 遍历远程目录
            completed_files = 0
            stdin, stdout, stderr = ssh.exec_command(f'find {self.source_dir} -type f')
            remote_files = stdout.read().decode().splitlines()
            
            for remote_file in remote_files:
                relative_path = Path(remote_file).relative_to(self.source_dir)
                local_path = self.target_dir / relative_path
                
                if not self.transfer_file(remote_file, str(local_path)):
                    self.logger.error(f"传输失败，停止传输: {remote_file}")
                    return False
                
                completed_files += 1
                
                # 添加短暂延迟，避免系统负载过高
                time.sleep(0.1)
            
            # 关闭进度条
            if self.pbar:
                self.pbar.close()
            
            # 显示最终统计信息
            elapsed_time = time.time() - self.start_time
            # 使用self.total_size来计算平均速度，因为这是我们在开始时统计的总大小
            average_speed = self.total_size / elapsed_time if elapsed_time > 0 else 0
            print(f"\n传输完成！总耗时: {elapsed_time:.1f}秒 平均速度: {self._format_size(average_speed)}/s")
            return True
            
        except Exception as e:
            self.logger.error(f"传输目录失败: {str(e)}")
            return False
        finally:
            try:
                ssh.close()
            except:
                pass

if __name__ == "__main__":
    # 使用示例
    transfer = FileTransfer(
        source_dir="",
        target_dir="",
        hostname="",
        port=,
        username="",
        password="",
        log_level=logging.INFO,  # 设置日志级别
        verify_existing=True     # 是否验证已存在的文件
    )
    transfer.transfer_directory()
