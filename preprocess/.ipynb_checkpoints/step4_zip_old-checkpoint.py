import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    ScaleIntensityRanged
)
from monai.data import DataLoader, Dataset
from typing import List, Tuple, Union
import SimpleITK as sitk
import pickle
import nibabel as nib
import json
import numpy as np
import os
def _sample_foreground_locations(seg: np.ndarray, classes_or_regions: Union[List[int], List[Tuple[int, ...]]],
                                     seed: int = 1234, verbose: bool = False):
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too
        # sparse
        rndst = np.random.RandomState(seed)
        class_locs = {}
        for c in classes_or_regions:
            k = c if not isinstance(c, list) else tuple(c)
            if isinstance(c, (tuple, list)):
                mask = seg == c[0]
                for cc in c[1:]:
                    mask = mask | (seg == cc)
                all_locs = np.argwhere(mask)
            else:
                all_locs = np.argwhere(seg == c)
            if len(all_locs) == 0:
                class_locs[k] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[k] = selected[:, 1:]
            if verbose:
                print(c, target_num_samples)
        return class_locs

def read_nii_head(nifti_file):
    nifti_img = nib.load(nifti_file)
    # 获取头信息
    #header = nifti_img.header
    # 获取数据数组的维度
    data_shape = nifti_img.header.get_data_shape()
    # 获取三维像素尺寸（XYZ）
    pixel_sizes = nifti_img.header['pixdim'][1:4]  # 提取第2、3、4个元素，即X、Y、Z轴的像素尺寸
    return(data_shape,pixel_sizes)


def preprocess(data_dic,preprocess_folder):
    
    total_transformd = Compose([
        # 加载图像，会默认根据文件后缀选择相应的读取类
        LoadImaged(keys=["image", "semi_auto_infor", "label"]),
        # 根据前景裁剪，会把前景部分裁剪出来
        CropForegroundd(keys=["image", "semi_auto_infor", "label"], source_key="image", margin=4, allow_smaller=True),
        #限制接受图像的范围
        ScaleIntensityRanged(keys=["image"], a_min=-255, a_max=255.0, b_min=0.0, b_max=1, clip=True),
        #ScaleIntensityRanged(keys=["semi_auto_infor"], a_min=0, a_max=200.0, b_min=0.0, b_max=1, clip=True),
        # 增加通道维度
        EnsureChannelFirstd(keys=["image", "semi_auto_infor","label"])
])
    dataset = Dataset(data=data_dic, transform=total_transformd)
    
    # 提取每个样本的图像和标签数据
    images_list = [sample['image'] for sample in dataset]
    semi_list = [sample["semi_auto_infor"] for sample in dataset]
    label_list = [sample['label'] for sample in dataset]
    
    
    #返回保存的内容
    data_keys = []
    data_paths = []
    pkl_paths = []

    for i in range(len(images_list)):
        num = len(data_dic[i]['image'].split('/'))
        save_path = preprocess_folder +'/'+ data_dic[i]['image'].split('/')[num-1][:-7] + '.npz'
        np.savez_compressed(save_path, image=images_list[i], semi_auto=semi_list[i], label=label_list[i])
        
        # 使用集合获取不重复的字符
        unique_chars = label_list[i].unique().tolist()
        #print(unique_chars)
        properties = {}
        
        properties['original_size_of_raw_data'],properties['original_spacing'] = read_nii_head(data_dic[i]['label'])
        properties['size_after_cropping'] = images_list[i].shape
        properties['classes'] = unique_chars

        collect_for_this = [x for x in unique_chars if x != 0.0]
        properties['class_locations'] = _sample_foreground_locations(label_list[i], collect_for_this, verbose=True)
        #print(properties['class_locations'])
        
        print(properties['class_locations'])
        #print(properties)
        print(data_dic[i]['image'].split('/')[num-1][:-7] + ' has finnished preprocess.')
        # 保存properties为.pkl文件
        pkl_file_path = preprocess_folder +'/'+ data_dic[i]['image'].split('/')[num-1][:-7] + '.pkl'
        #print(pkl_file_path)
        with open(pkl_file_path, "wb") as f:
            pickle.dump(properties, f)
        data_keys.append(data_dic[i]['image'].split('/')[num-1][:-7])
        data_paths.append(preprocess_folder +'/'+data_dic[i]['image'].split('/')[num-1][:-7] + '.npz')
        pkl_paths.append(preprocess_folder +'/'+ data_dic[i]['image'].split('/')[num-1][:-7] + '.pkl')
    return(data_keys,data_paths,pkl_paths)


def run_zip(data_summary_json_path: str, preprocess_folder: str, output_json_path: str):
    """
    执行预处理逻辑

    参数:
        data_summary_json_path (str): 数据汇总 JSON 文件的路径
        preprocess_folder (str): 预处理输出文件夹路径
        output_json_path (str): 输出 JSON 文件路径
    """
    # 加载配置文件，获取图像和标签路径
    #print('step2')
    with open(data_summary_json_path, 'r') as f:
        config = json.load(f)
    data_dic = config['data']

    # 初始化 JSON 数据结构
    json_data = {
        'keys': [],
        'data_paths': [],
        'pkl_paths': []
    }

    # 创建预处理输出文件夹
    os.makedirs(preprocess_folder, exist_ok=True)

    # 分批处理数据
    for i in range(0, len(data_dic), 2):
        # 创建小批次数据
        small_batch = [data_dic[i]]
        if i + 1 < len(data_dic):
            small_batch.append(data_dic[i + 1])

        # 调用预处理函数
        data_keys, data_paths, pkl_paths = preprocess(small_batch, preprocess_folder)

        # 将结果添加到 JSON 数据中
        json_data['keys'].append(data_keys)
        json_data['data_paths'].append(data_paths)
        json_data['pkl_paths'].append(pkl_paths)

    # 保存 JSON 文件
    with open(output_json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
