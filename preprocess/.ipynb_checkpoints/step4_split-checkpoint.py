import json
import glob
import os
import random

def split_train_data(folder):
    output_folder = folder
    os.makedirs(output_folder, exist_ok=True)

    # 加载原始 JSON 文件
    with open(output_folder+'/data_preprocess.json', 'r') as f:
        original_data = json.load(f)

    # 假设原始数据中包含 keys, data_paths, pkl_paths
    keys = original_data['keys']
    data_paths = original_data['data_paths']
    pkl_paths = original_data['pkl_paths']

    # 进行五折交叉验证数据划分
    num_samples = len(keys)
    fold_size = num_samples // 5

    for fold in range(5):
        start_idx = fold * fold_size
        end_idx = (fold + 1) * fold_size

        val_keys = keys[start_idx:end_idx]
        val_data_paths = data_paths[start_idx:end_idx]
        val_pkl_paths = pkl_paths[start_idx:end_idx]

        train_keys = keys[:start_idx] + keys[end_idx:]
        train_data_paths = data_paths[:start_idx] + data_paths[end_idx:]
        train_pkl_paths = pkl_paths[:start_idx] + pkl_paths[end_idx:]

        fold_data = {
            'train_keys': train_keys,
            'train_data_paths': train_data_paths,
            'train_pkl_paths': train_pkl_paths,
            'val_keys': val_keys,
            'val_data_paths': val_data_paths,
            'val_pkl_paths': val_pkl_paths
        }

        # 写入新的 JSON 文件至指定文件夹
        output_file_path = os.path.join(output_folder, f'fold_{fold}.json')
        with open(output_file_path, 'w') as outfile:
            json.dump(fold_data, outfile, indent=4)