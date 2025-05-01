import pandas as pd
import numpy as np
import nibabel as nib
from scipy import ndimage
import os


# 读取Excel文件
def get_points(file_path, target_string, pre=True):
    data = pd.read_excel(file_path)
    # 检查是否存在"leision"列
    if "leision" in data.columns:
        leision_content = data["leision"].tolist()
        found_rows = []
        for index, content in enumerate(leision_content):
            if target_string in content:
                found_rows.append(index)
                # print(found_rows)
        points = []
        if found_rows:
            # print(f"包含'{target_string}'的行中，'x_pre'列的元素:")
            for row in found_rows:
                if pre:
                    x_pre_value = data.loc[row, "x_pre"]
                    y_pre_value = data.loc[row, "y_pre"]
                    z_pre_value = data.loc[row, "z_pre"]
                    #points.append((int(z_pre_value), int(y_pre_value),int(x_pre_value)))
                    points.append((int(z_pre_value), int(y_pre_value), int(x_pre_value)))
                else:
                    x_value = data.loc[row, "x"]
                    y_value = data.loc[row, "y"]
                    z_value = data.loc[row, "z"]
                    #points.append((int(z_value), int(y_value), int(x_value)))
                    points.append((int(z_value), int(y_value), int(x_value)))
    return points


import SimpleITK as sitk
import numpy as np


def delete_neg_positive(input_file, points, output_file):
    # 加载 NIfTI 图像数据
    img = sitk.ReadImage(input_file)
    data = sitk.GetArrayFromImage(img)
    #print(data.shape)
    # 定义多个目标点的坐标
    target_points = points

    # 找到所有的联通区域
    labeled, num_features = ndimage.label(data)

    # 创建一个新的数组用于存储符合条件的数据
    new_data = np.zeros_like(data)

    # 遍历所有的目标点
    for target_point in target_points:
        x, y, z = target_point
        target_label = labeled[x, y, z]

        # 遍历所有的联通区域
        for label in range(1, num_features + 1):
            # 如果目标点在这个联通区域内，保留这个区域
            if np.any(labeled == label) and label == target_label:
                new_data[labeled == label] = 1

    # 对处理后的数据进行类型转换为 uint8
    new_data_uint8 = new_data.astype(np.uint8)

    # 创建一个新的 SimpleITK 图像对象并设置头信息
    new_img_uint8 = sitk.GetImageFromArray(new_data_uint8)
    new_img_uint8.CopyInformation(img)

    # 保存处理后的图像数据为 uint8 类型
    sitk.WriteImage(new_img_uint8, output_file)

# file_path = "E:/Scientific_work/1Liver_cancer/1Liver03data/6Task103/cancer_infor_cutpage_big_tumour_test - 副本.xlsx"  # 请将"your_file_path.xlsx"替换为实际的Excel文件路径
# target_string = "liver_102"
# points = get_points(file_path, target_string)
# print(points)
# delete_neg_positive('C:/Users/13603/Desktop/liver_102.nii.gz',points,'C:/Users/13603/Desktop/liver_102_p.nii.gz')

def postprocess_main(input_fold, output_fold, xlxs_path):
    if not os.path.exists(output_fold):
        os.makedirs(output_fold)

    def list_files_in_directory(input_fold):
        files_list = []
        for root, dirs, files in os.walk(input_fold):
            for file_name in files:
                if file_name.endswith('.nii.gz'):
                    files_list.append(file_name)
        return files_list

    files_in_folder = list_files_in_directory(input_fold)

    total_files = len(files_in_folder)

    for i, file in enumerate(files_in_folder):
        input_file = os.path.join(input_fold, file)
        output_file = os.path.join(output_fold, file)
        points = get_points(xlxs_path, target_string=file[:-7])
        #print(input_file, points)
        delete_neg_positive(input_file, points, output_file)

        # 手动显示进度
        progress = (i + 1) / total_files * 100
        print(f"Progress: {progress:.2f}% ({i + 1}/{total_files} files processed)")


#postprocess_main("/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/MSDliver/result/fold_1/predict/1", 
#            "/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/MSDliver/result/fold_1/predict/post_1", 
#            "/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/MSDliver/data/cancer_infor_cutpage_big_tumour_test.xlsx")