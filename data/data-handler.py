import os
import tarfile
import shutil

# 定义路径
base_dir = './'
output_dirs = {
    'train': os.path.join(base_dir, 'train'),
    'test': os.path.join(base_dir, 'test')
}

# 创建train和test文件夹（如不存在）
for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# 遍历所有的tar.gz文件
for file_name in os.listdir('../dataset/wav'):
    if file_name.endswith('.tar.gz'):
        tar_path = os.path.join('../dataset/wav', file_name)
        base_name = file_name.replace('.tar.gz', '')  # S0002-S0916

        # 创建一个临时解压目录
        temp_dir = os.path.join(base_dir, 'temp_extract')
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir)

        # 解压 tar.gz 文件
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)

        # 解压后结构为 temp_dir/train/S0002 或 temp_dir/test/S0002
        for split in ['train', 'test']:
            split_path = os.path.join(temp_dir, split)
            if not os.path.exists(split_path):
                continue

            sub_dir = os.path.join(split_path, base_name)
            if not os.path.exists(sub_dir):
                continue

            # 遍历音频文件
            for fname in os.listdir(sub_dir):
                if fname.lower().endswith('.wav'):
                    src_file = os.path.join(sub_dir, fname)
                    new_fname = f'{base_name}-{split}-{base_name}-{fname}'
                    dst_file = os.path.join(output_dirs[split], new_fname)
                    shutil.move(src_file, dst_file)

        # 清理临时目录
        shutil.rmtree(temp_dir)

print("数据整理完成！")
