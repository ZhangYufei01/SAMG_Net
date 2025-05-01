import inspect
import itertools
import multiprocessing
from multiprocessing import Pool
import os
from copy import deepcopy
from time import sleep
from typing import Tuple, Union, List, Optional
import torch
from tqdm import tqdm
from Model_3DU_SE import UNet
from typing import Tuple, Union, List, Optional, Dict, Any
import numpy as np
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    ToTensord,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    RandRotated
)
from monai.data import DataLoader, Dataset
from related_to_software.simpleitk_reader_writer import SimpleITKIO
import json
from related_to_software.predict_data_iterators import preprocessing_iterator_fromnpy,preprocessing_iterator_fromfiles
from related_to_hardware.predict_empty_cache import empty_cache,dummy_context
from related_to_software.predict_padding import pad_nd_image
from related_to_software.predict_sliding_window import compute_gaussian,compute_steps_for_sliding_window
from related_to_software.predict_export import export_prediction_from_logits,convert_predicted_logits_to_segmentation_with_correct_shape
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
import matplotlib.pyplot as plt
import psutil

torch._dynamo.config.cache_size_limit = 64

class Unet3D_Predictor(object):
    def __init__(self,
                 dataset_json: dict,
                 patch_size : tuple,
                 input_channels: int,
                 output_clannels: int,
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 perform_everything_on_device: bool = True,
                 device: torch.device = torch.device('cuda'),
                 verbose: bool = False,
                 verbose_preprocessing: bool = False,
                 allow_tqdm: bool = True
                 ):
        self.input_channels = input_channels
        self.output_clannels = output_clannels
        self.dataset_json = dataset_json
        self.verbose = verbose
        self.verbose_preprocessing = verbose_preprocessing
        self.allow_tqdm = allow_tqdm

        self.plans_manager, self.configuration_manager, self.list_of_parameters, self.network, self.dataset_json, \
        self.trainer_name, self.allowed_mirroring_axes, self.label_manager = None, None, None, None, None, None, None, None

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.default_num_processes=8
        if device.type == 'cuda':
            # device = torch.device(type='cuda', index=0)  # set the desired GPU with CUDA_VISIBLE_DEVICES!
            pass
        if device.type != 'cuda':
            print(f'perform_everything_on_device=True is only supported for cuda devices! Setting this to False')
            perform_everything_on_device = False
        self.device = device
        self.perform_everything_on_device = perform_everything_on_device
        self.patch_size = patch_size

    def initialize_from_trained_model_folder(self, checkpoint_file_path='E:/Scientific_work/1Liver_cancer/1Liver03data/6Task103/output_data/checkpoint_latest.pth'
                                             ):
        """
        This is used when making predictions with a trained model
        """

        parameters = []
        #for i, f in enumerate(use_folds):
         #   f = int(f) if f != 'all' else f
            #print(model_training_output_dir+'/fold_', f+'_result/'+ checkpoint_name)
        checkpoint = torch.load(checkpoint_file_path,map_location=torch.device('cpu'))

        trainer_name = checkpoint['trainer_name']
        #configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
                'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        parameters.append(checkpoint['network_weights'])
        #print(f'inference_allowed_mirroring_axes{inference_allowed_mirroring_axes}')
        #network = UNet3D_concat_SE(in_channels=self.input_channels, out_channels=self.output_clannels, bilinear=True)
        network = UNet(in_channels=self.input_channels, out_channels=self.output_clannels)
        self.list_of_parameters = parameters
        self.network = network.to(self.device)
        self.network = torch.compile(self.network)
        self.trainer_name = trainer_name
        self.allowed_mirroring_axes = inference_allowed_mirroring_axes
        

    '''def read_images_from_paths(self,image_paths):
        print(image_paths)
        imgs = []
        props = []
        for path in image_paths:
            img, prop = SimpleITKIO().read_images(path)
            imgs.append(img)
            props.append(prop)
            print(props)
        return imgs, props'''

    def read_images_from_paths(self, image_path,semi_path):

        simple_itk_io = SimpleITKIO()

        img, prop = simple_itk_io.read_images(image_path)
        #print(img.shape)
        semi,prop_no_use = simple_itk_io.read_images(semi_path)

        return img, semi, prop



    def get_data_iterator_from_raw_npy_data(self,
                                            image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                                            segs_from_prev_stage_or_list_of_segs_from_prev_stage: Union[None,np.ndarray,List[np.ndarray]],
                                            semi_list:Union[np.ndarray, List[np.ndarray]],
                                            properties_or_list_of_properties: Union[dict, List[dict]],
                                            truncated_ofname: Union[str, List[str], None],
                                            num_processes: int = 3):
        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else \
            image_or_list_of_images
        list_of_semis = [semi_list] if not isinstance(semi_list, list) else \
            semi_list
        if isinstance(segs_from_prev_stage_or_list_of_segs_from_prev_stage, np.ndarray):
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = [
                segs_from_prev_stage_or_list_of_segs_from_prev_stage]
        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]
        if isinstance(properties_or_list_of_properties, dict):
            properties_or_list_of_properties = [properties_or_list_of_properties]
        num_processes = min(num_processes, len(list_of_images))
        pp = preprocessing_iterator_fromnpy(
            list_of_images,
            segs_from_prev_stage_or_list_of_segs_from_prev_stage,
            list_of_semis,
            properties_or_list_of_properties,
            truncated_ofname,
            num_processes,
            self.verbose_preprocessing
        )

        return pp

    def check_workers_alive_and_busy(self,export_pool: Pool, worker_list: List, results_list: List,
                                     allowed_num_queued: int = 0):
        """

        returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
        """
        alive = [i.is_alive() for i in worker_list]
        if not all(alive):
            raise RuntimeError('Some background workers are no longer alive')

        not_ready = [not i.ready() for i in results_list]
        if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
            return True
        return False

    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.patch_size) < len(image_size):
            assert len(self.patch_size) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {self.patch_size}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.patch_size)]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.patch_size,
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {self.patch_size}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.patch_size)]]))
        return slicers

    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        #print(x.shape)
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        #X_data = x[][0].unsqueeze(0)
        #X_semi = x[1].unsqueeze(0)
        #split_x = torch.split(x, 1, dim=1)
        #print(x.shape)
        prediction = self.network(x)
        #print(f'in predict main, mirror_axes:{mirror_axes}')
        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations([m + 2 for m in mirror_axes], i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, (*axes,))), (*axes,))
                #flipped_x_0 = torch.flip(split_x[0], (*axes,))
                #flipped_x_1 = torch.flip(split_x[1], (*axes,))
                #prediction += torch.flip(self.network(flipped_x_0,flipped_x_1), (*axes,))
            prediction /= (len(axes_combinations) + 1)
        return prediction

    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            '''image_to_show = data[0, :, :, 20].cpu().numpy()  # 选择第一个图像
            plt.figure(figsize=(6, 6))
            plt.imshow(image_to_show, cmap='gray')  # 显示灰度图像
            plt.axis('off')  # 关闭坐标轴
            plt.show()'''


            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            #self.label_manager.num_segmentation_heads负责输出要分割的数量（指定的前景区域数量/标签数量），直接换成通道输出数
            predicted_logits = torch.zeros((self.output_clannels, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.patch_size), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)

            if self.verbose: print('running prediction')
            if not self.allow_tqdm and self.verbose: print(f'{len(slicers)} steps')
            #print(f'slicer{slicers}')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device, non_blocking=False)
                #print(workon.shape)
                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                predicted_logits[sl] += (prediction * gaussian if self.use_gaussian else prediction)
                n_predictions[sl[1:]] += (gaussian if self.use_gaussian else 1)

            predicted_logits /= n_predictions


            '''image_to_show = predicted_logits[0, 0, :, :, 20].cpu().numpy()  # 选择第一个图像
            plt.figure(figsize=(6, 6))
            plt.imshow(image_to_show, cmap='gray')  # 显示灰度图像
            plt.axis('off')  # 关闭坐标轴
            plt.show()'''



            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits

    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        assert isinstance(input_image, torch.Tensor)
        self.network = self.network.to(self.device)
        self.network.eval()

        empty_cache(self.device)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
        # and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
        # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with torch.no_grad():
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():

                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose: print(f'Input shape: {input_image.shape}')
                if self.verbose: print("step_size:", self.tile_step_size)
                if self.verbose: print("mirror_axes:", self.allowed_mirroring_axes if self.use_mirroring else None)

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.patch_size,
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                               self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[tuple([slice(None), *slicer_revert_padding[1:]])]
        return predicted_logits
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """

        n_threads = torch.get_num_threads()
        torch.set_num_threads(self.default_num_processes if self.default_num_processes < n_threads else n_threads)
        with torch.no_grad():
            prediction = None

            for params in self.list_of_parameters:

                # messing with state dict names...

                self.network.load_state_dict(params)

                # why not leave prediction on device if perform_everything_on_device? Because this may cause the
                # second iteration to crash due to OOM. Grabbing that with try except cause way more bloated code than
                # this actually saves computation time
                if prediction is None:
                    prediction = self.predict_sliding_window_return_logits(data).to('cpu')
                else:
                    prediction += self.predict_sliding_window_return_logits(data).to('cpu')

            if len(self.list_of_parameters) > 1:
                prediction /= len(self.list_of_parameters)

            if self.verbose: print('Prediction done')
            prediction = prediction.to('cpu')
        torch.set_num_threads(n_threads)
        return prediction

    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 1):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        
            
            
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in data_iterator:

                
                data = preprocessed['data']
                #print('line384',data.shape)
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                #print('ofile',ofile)
                if ofile is not None:
                    print(f'\nPredicting {os.path.basename(ofile)}:')
                else:
                    print(f'\nPredicting image of shape {data.shape}:')

                print(f'perform_everything_on_device: {self.perform_everything_on_device}')

                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not self.check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not self.check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                #print(data.shape)
                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if ofile is not None:
                    # this needs to go into background processes
                    # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                    #                               self.dataset_json, ofile, save_probabilities)
                    print('sending off prediction to background worker for resampling and export')
                    r.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, ofile, save_probabilities),)
                        )
                    )
                else:
                    # convert_predicted_logits_to_segmentation_with_correct_shape(
                    #             prediction, self.plans_manager,
                    #              self.configuration_manager, self.label_manager,
                    #              properties,
                    #              save_probabilities)

                    print('sending off prediction to background worker for resampling')
                    r.append(
                        export_pool.starmap_async(
                            convert_predicted_logits_to_segmentation_with_correct_shape, (
                                (prediction, properties,save_probabilities),)
                        )
                    )
                if ofile is not None:
                    print(f'done with {os.path.basename(ofile)}')
                else:
                    print(f'\nDone with image of shape {data.shape}:')
            ret = [i.get()[0] for i in r]

        if isinstance(data_iterator, MultiThreadedAugmenter):
            data_iterator._finish()

        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret


    def predict_from_files(self,
                           image_or_list_of_images: Union[np.ndarray, List[np.ndarray]],
                           semi_list: Union[np.ndarray, List[np.ndarray]],
                           truncated_ofname: Union[str, List[str], None],
                           intensityproperties:Dict[str, Any],
                           save_probabilities: bool = False,
                           num_processes_preprocessing: int = 1,
                           num_processes_segmentation_export: int = 1,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        """
        This is nnU-Net's default function for making predictions. It works best for batch predictions
        (predicting many images at once).
        """
        list_of_images = [image_or_list_of_images] if not isinstance(image_or_list_of_images, list) else image_or_list_of_images
        list_of_semis = [semi_list] if not isinstance(semi_list, list) else semi_list
        # let's store the input arguments so that its clear what was used to generate the prediction
        if isinstance(truncated_ofname, str):
            truncated_ofname = [truncated_ofname]

        num_processes = min(num_processes_preprocessing, len(list_of_images))
        my_init_kwargs = {}
        for k in inspect.signature(self.predict_from_files).parameters.keys():
            my_init_kwargs[k] = locals()[k]
        my_init_kwargs = deepcopy(my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a

        data_iterator = self._internal_get_data_iterator_from_lists_of_filenames(list_of_images,
                                                                                 list_of_semis,
                                                                                 truncated_ofname,
                                                                                 intensityproperties,
                                                                                 num_processes)

        return self.predict_from_data_iterator(data_iterator, save_probabilities, num_processes_segmentation_export)

    def _internal_get_data_iterator_from_lists_of_filenames(self,
                                                            input_list_of_lists: List[List[str]],
                                                            input_list_of_semis: List[List[str]],
                                                            output_filenames_truncated: Union[List[str], None],
                                                            intensityproperties:Dict[str, Any],
                                                            num_processes: int):
        return preprocessing_iterator_fromfiles(input_list_of_lists, input_list_of_semis,
                                                output_filenames_truncated,intensityproperties,num_processes, self.device.type == 'cuda',
                                                self.verbose_preprocessing)



