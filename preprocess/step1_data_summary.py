import json
import glob
import os
import random


def get_data_summary(data_folder,image_key='imagesTr',label_key='labelsTr',semi_key='semiTr'):
    data_summary = {"data": []}
    image_folder = data_folder +'/' + image_key
    label_folder = data_folder +'/' + label_key
    semi_folder = data_folder +'/' + semi_key
    #print(image_folder)
    file_list = os.listdir(image_folder)
    for file in file_list:
        #print(file)
        image = image_folder +'/' + file
        label = label_folder +'/' + file
        semi_auto_infor = semi_folder +'/' + file[:-7] + '_star_rain.nii.gz'
        if os.path.exists(label) and os.path.exists(semi_auto_infor):
            data_entry = {
                "image": image,
                "label": label,
                "semi_auto_infor": semi_auto_infor
            }
            data_summary["data"].append(data_entry)
    # Save the data summary to a JSON file
    with open(data_folder+"/data_summary.json", "w") as f:
        json.dump(data_summary, f, indent=4)
        
#get_data_summary('/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/Beigene_liver/data')