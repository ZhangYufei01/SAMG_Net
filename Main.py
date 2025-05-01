import argparse
import json
import torch
from semi_Train_Main import Unet3D_Train
from predict_Main import Unet3D_Predictor
from postprocess_delete_false_positive import postprocess_main
from evaluation_Main import compute_metrics_on_folder, labels_to_list_of_regions
from batchgenerators.utilities.file_and_folder_operations import isdir, maybe_mkdir_p
from related_to_software.simpleitk_reader_writer import SimpleITKIO
from preprocess.step1_data_summary import get_data_summary 
from preprocess.step2_read_fingerprint import process_dataset
from preprocess.step3_calculate import DefaultPreprocessor
from preprocess.step4_split import split_train_data
from preprocess.step5_predict import predict_json

def train(data_infor_json_path, output_folder, preprocessed_dataset_folder, fold, patch_size):
    #print(data_infor_json_path, output_folder, preprocessed_dataset_folder, fold, patch_size)
    with open(data_infor_json_path, 'r') as f:
        json_data = json.load(f)
    Train = Unet3D_Train(input_clannels=2, output_clannels=2, output_folder=output_folder, patch_size=patch_size, batch_size=2, preprocessed_dataset_folder=preprocessed_dataset_folder, annotated_classes_key=[1], fold=fold, dataset_json=json_data)
    Train.run_training()

def predict(data_infor_json_path, output_folder, fold, checkpoint, patch_size):
    with open(data_infor_json_path, 'r') as f:
        json_data = json.load(f)

    image_paths = [entry['image'] for entry in json_data['predict_filenames']]
    semi_paths = [entry['semi'] for entry in json_data['predict_filenames']]
    output_folder_fold = output_folder + '/' + str(fold)
    names = [entry['name'] for entry in json_data['predict_filenames']]
    
    intensityproperties = json_data['intensity_statistics_per_channel']

    if not isdir(output_folder_fold):
        maybe_mkdir_p(output_folder_fold)
    output_paths = [output_folder_fold + '/' + name for name in names]

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")

    predictor = Unet3D_Predictor(
        dataset_json=json_data,
        patch_size=patch_size,
        input_channels=2,
        output_clannels=2,
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    predictor.initialize_from_trained_model_folder(
        checkpoint_file_path=checkpoint
    )

    predictor.predict_from_files([image_paths],
                                 [semi_paths],
                                 [output_paths],
                                 intensityproperties,
                                 save_probabilities=False,
                                 num_processes_preprocessing=1, num_processes_segmentation_export=1,
                                 folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0)


def postprocess(input_folder,output_folder,xlxs_path):
    postprocess_main(input_folder,output_folder,xlxs_path)

def evaluation(GT_folder,output_folder,json_file):
    image_reader_writer = SimpleITKIO()
    file_ending = '.nii.gz'
    #regions = labels_to_list_of_regions([1,2,3,4,5,6])
    regions = labels_to_list_of_regions([1])
    ignore_label = None
    num_processes = 3
    compute_metrics_on_folder(GT_folder, output_folder, json_file, image_reader_writer, regions, file_ending,  ignore_label,
                              num_processes)

def preprocess_all(folder):
    #print('in preprocess')
    print(f'Manipulate {folder}')
    get_data_summary(folder)
    json_file = folder+'/'+'data_summary.json'
    process_dataset(json_file)
    pp = DefaultPreprocessor()
    preprocess_folder = folder+'/'+"preprocess_data"
    preprocess_json =  folder+'/'+"data_preprocess.json"
    pp.run(json_file, preprocess_folder,preprocess_json,3)
    split_train_data(folder)
    predict_json(folder)
    
def tuple_arg(s):
    try:
        return tuple(map(int, s.strip('()').split(',')))
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid tuple format: {s}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or predict using Unet3D model")
    parser.add_argument("--m", type=str, choices=["train", "predict","postprocess","evaluation","preprocess"], help="Select mode")
    parser.add_argument("--d", type=str, help="Path to the data information JSON file")
    parser.add_argument("--o", type=str, help="Path to the output folder")
    parser.add_argument("--pd", type=str, help="Path to the preprocessed dataset folder")
    parser.add_argument("--f", type=int, help="Fold value", default=0)
    parser.add_argument("--p", type=lambda x: tuple_arg(x), help="Patch size", default=(192, 192, 64))
    parser.add_argument("--c", type=str, help="Path of the checkpoint")
    parser.add_argument("--ip", type=str, help="Path of the input predicted folder to do postprocess")
    parser.add_argument("--op", type=str, help="Path of the onput folder after postprocess")
    parser.add_argument("--x", type=str, help="Path of the xlxs file")
    parser.add_argument("--g", type=str, help="Path of the ground_truth folder")
    parser.add_argument("--j", type=str, help="Path of the result, save in .json file")
    parser.add_argument("--pre", type=str, help="Path of the folder for preprocessing")
    args = parser.parse_args()

    if args.m == "train":
        train(data_infor_json_path=args.d, output_folder=args.o, preprocessed_dataset_folder=args.pd, fold=args.f, patch_size=args.p)
    elif args.m == "predict":
        predict(data_infor_json_path=args.d, output_folder=args.o, fold=args.f, checkpoint = args.c, patch_size=args.p)
    elif args.m == "postprocess":
        postprocess(input_folder=args.ip,output_folder=args.op,xlxs_path=args.x)
    elif args.m == "evaluation":
        evaluation(GT_folder=args.g,output_folder=args.op,json_file=args.j)
    elif args.m == "preprocess":
        preprocess_all(folder = args.pre)