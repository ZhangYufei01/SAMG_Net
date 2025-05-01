import os
import json



def predict_json(folder):
    summary_filename = folder + '/data_summary.json'
    config_filename = folder + '/predict_semi.json'
    images=folder + '/imagesTs'
    labels=folder + '/labelsTs'
    semis=folder + '/semiTs'
    semi_key = '_star_rain'
    with open(summary_filename, 'r') as f:
        summary = json.load(f)

    # 获取文件夹中所有文件的文件名
    file_names = os.listdir(images)
    print(file_names)

    config = dict()
    config["predict_filenames"] = []
    config['intensity_statistics_per_channel'] = summary['intensity_statistics_per_channel']
    for image_name in file_names:
        if image_name.endswith('.nii.gz'):
            image_path = images + '/' + image_name
            label_path = labels + '/' + image_name
            if os.path.exists(label_path):
                semi_path = semis + '/' + image_name[:-7] + semi_key + '.nii.gz'
                if os.path.exists(semi_path):
                    config["predict_filenames"].append({"image": image_path, "label": label_path, "semi":semi_path, "name":image_name})
                else:

                    print(f'Image_name:{image_name} can not find a match semi_information file in folder:{semis}. ')
            else:
                print(f'Image_name:{image_name} can not find a match label file in folder:{labels}. ')


    with open(config_filename, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Generated config file: {config_filename}")
