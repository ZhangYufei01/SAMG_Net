import torch
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, Resize,Lambda
import numpy as np
import random
from PIL import Image
from scipy.ndimage import zoom
#from Model_semi_NestedUNet_se import NestedUNet
from Model_3DU_SE import UNet
from time import time, sleep
from monai.losses import DiceCELoss,DiceLoss
from datetime import datetime
import os
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
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, \
    ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
from batchgenerators.transforms.utility_transforms import RemoveLabelTransform, RenameTransform, NumpyToTensor
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from typing import Union, Tuple, List
from dataloading.nnUDataset import nnUNetDataset
from dataloading.nnUDataloader import nnUNetDataLoader3D
from related_to_software.Logging import Logger
from related_to_software.learning_rate_scheduler import PolyLRScheduler
from related_to_hardware.dummy_context import dummy_context
from related_to_hardware.default_n_proc_DA import get_allowed_n_proc_DA
from related_to_software.evaluation_unit import get_tp_fp_fn_tn
import sys
from torch import autocast, nn
from torch.cuda.amp import GradScaler
from related_to_software.collate_outputs import collate_outputs
import torch.nn.functional as F
#from torch._dynamo import OptimizedModule
import inspect
from loss.Dice_CE_loss import DC_and_CE_loss
from loss.Diceloss import MemoryEfficientSoftDiceLoss
class Unet3D_Train(object):
    def __init__(self, input_clannels:int, output_clannels:int, output_folder:str, patch_size:tuple, batch_size:int, preprocessed_dataset_folder:str,annotated_classes_key:list, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):

        self.device = device
        self.input_clannels = input_clannels
        self.output_clannels = output_clannels
        ###  Saving all the init args into class variables for later access
        self.output_folder = output_folder
        self.patch_size = patch_size
        self.batch_size = batch_size
        #self.configuration_manager = self.plans_manager.get_configuration(configuration)
        #self.configuration_name = configuration
        self.dataset_json = dataset_json
        self.fold = fold
        self.unpack_dataset = unpack_dataset
        self.preprocessed_dataset_folder = preprocessed_dataset_folder
        self.annotated_classes_key = annotated_classes_key #想要的前景类
        # unlike the previous nnunet folder_with_segs_from_previous_stage is now part of the plans. For now it has to
        # be a different configuration in the same plans
        # IMPORTANT! the mapping must be bijective, so lowres must point to fullres and vice versa (using
        # "previous_stage" and "next_stage"). Otherwise it won't work!
        self.is_cascaded = None


        ### Some hyperparameters for you to fiddle with
        self.initial_lr = 1e-4
        self.weight_decay = 3e-7
        self.oversample_foreground_percent = 0.33
        self.num_iterations_per_epoch = 250
        #++++++++++++++++++++++++++++++这里记得改回去++++++++++++++++++++++++++++++++
        #self.num_iterations_per_epoch = 10
        self.num_val_iterations_per_epoch = 50
        #self.num_val_iterations_per_epoch = 5
        self.num_epochs = 500
        self.current_epoch = 0
        self.enable_deep_supervision = True
        self.logger = Logger()
        self.local_rank = 0 #这个表示当前是主进程（唯一进程）
        ### Dealing with labels/regions
        #self.label_manager = self.plans_manager.get_label_manager(dataset_json)
        # labels can either be a list of int (regular training) or a list of tuples of int (region-based training)
        # needed for predictions. We do sigmoid in case of (overlapping) regions
        timestamp = datetime.now()
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        self.num_input_channels = None  # -> self.initialize()
        self.network = None  # -> self.build_network_architecture()
        self.optimizer = self.lr_scheduler = None  # -> self.initialize
        self.loss = None  # -> self.initialize
        self.grad_scaler = GradScaler() if self.device.type == 'cuda' else None
        ### Simple logging. Don't take that away from me!
        # initialize log file. This is just our log for the print statements etc. Not to be confused with lightning
        # logging
        timestamp = datetime.now()

        ### placeholders
        self.dataloader_train = self.dataloader_val = None  # see on_train_start

        ### initializing stuff for remembering things and such
        self._best_ema = None

        ### inference things
        self.inference_allowed_mirroring_axes = None  # this variable is set in
        # self.configure_rotation_dummyDA_mirroring_and_inital_patch_size and will be saved in checkpoints

        ### checkpoint saving stuff
        self.save_every = 50
        self.disable_checkpointing = False
        self.my_init_kwargs = {}
        for k in inspect.signature(self.__init__).parameters.keys():
            self.my_init_kwargs[k] = locals()[k]


    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,momentum=0.99, nesterov=True)
        '''optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-5, weight_decay=1e-5, amsgrad=True)  # 定义优化器
        return optimizer'''
        optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                    momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
    
    def setup_seed(self,seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        
    #初始化网络,优化器,loss信息
    def initialize(self):
        self.setup_seed(123)
        #model = UNet3D_concat_SE(in_channels=self.input_clannels, out_channels=self.output_clannels, bilinear=True)
        model = UNet(in_channels=self.input_clannels, out_channels=self.output_clannels)
        #print(f'line147,self.patch_size:{self.patch_size}')
        #model = UMambaEnc(input_size=self.patch_size, in_channels=2, out_channels=2)
        self.network =model.to(self.device)
        self.network = torch.compile(self.network)
        # compile network for free speedup
        self.optimizer, self.lr_scheduler = self.configure_optimizers()
        # if ddp, wrap in DDP wrapper
        #self.loss = DiceCELoss(to_onehot_y=True, weight=torch.tensor([0.1, 0.9]).to(self.device))
        self.loss = DC_and_CE_loss({'batch_dice': True,'smooth': 1e-5, 'do_bg': False, 'ddp': False}, {}, weight_ce=1, weight_dice=1,ignore_label=None, dice_class=MemoryEfficientSoftDiceLoss)

    #判断这个file是否存在，不存在就创建新目录
    def maybe_mkdir_p(self,directory: str) -> None:
        os.makedirs(directory, exist_ok=True)

    #释放缓存
    def empty_cache(self,device: torch.device):
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        else:
            pass

    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        if self.local_rank == 0:
            timestamp = time()
            dt_object = datetime.fromtimestamp(timestamp)

            if add_timestamp:
                args = (f"{dt_object}:", *args)

            successful = False
            max_attempts = 3
            ctr = 0
            while not successful and ctr < max_attempts:
                try:
                    with open(self.log_file, 'a+') as f:
                        for a in args:
                            f.write(str(a))
                            f.write(" ")
                        f.write("\n")
                    successful = True
                except IOError:
                    print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                    sleep(0.5)
                    ctr += 1
            if also_print_to_console:
                print(*args)
        elif also_print_to_console:
            print(*args)
    #用于计算最终的patch尺寸
    def get_patch_size(self,final_patch_size, rot_x, rot_y, rot_z, scale_range):
        if isinstance(rot_x, (tuple, list)):
            rot_x = max(np.abs(rot_x))
        if isinstance(rot_y, (tuple, list)):
            rot_y = max(np.abs(rot_y))
        if isinstance(rot_z, (tuple, list)):
            rot_z = max(np.abs(rot_z))
        rot_x = min(90 / 360 * 2. * np.pi, rot_x)
        rot_y = min(90 / 360 * 2. * np.pi, rot_y)
        rot_z = min(90 / 360 * 2. * np.pi, rot_z)
        from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
        coords = np.array(final_patch_size)
        final_shape = np.copy(coords)
        if len(coords) == 3:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        elif len(coords) == 2:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)
        final_shape /= min(scale_range)
        return final_shape.astype(int)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.patch_size
        dim = len(patch_size)

        rotation_for_DA = {
            'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
            'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
        }
        mirror_axes = (0, 1, 2)


        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = self.get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.85, 1.25))

        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, initial_patch_size, mirror_axes

    def get_training_transforms(self,
                                patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                mirror_axes: Tuple[int, ...],
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                ) -> AbstractTransform:
        tr_transforms = []

        patch_size_spatial = patch_size
        tr_transforms.append(SpatialTransform(
            patch_size_spatial, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,  # 这里是对他进行操作时候差值的维度，越大越平滑
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))
        tr_transforms.append(RemoveLabelTransform(-1, 0))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))
        tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                                   p_per_channel=0.5))
        tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
        tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                            p_per_channel=0.5,
                                                            order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                            ignore_axes=None))
        tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        #tr_transforms.append(RenameTransform('seg', 'target', True))

        tr_transforms.append(NumpyToTensor(['data', 'semi_infor', 'seg'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms

    def get_validation_transforms(self) -> AbstractTransform:
        val_transforms = []
        val_transforms.append(RemoveLabelTransform(-1, 0))
        #val_transforms.append(RenameTransform('seg', 'target', True))
        val_transforms.append(NumpyToTensor(['data', 'semi_infor','seg'], 'float'))
        val_transforms = Compose(val_transforms)
        return val_transforms

    def get_tr_and_val_datasets(self):
        # create dataset split
        tr_keys = self.dataset_json['train_keys']
        #print(tr_keys)
        val_keys = self.dataset_json['val_keys']
        #这里的keys就是preprocess文件夹里所有文件名去除.npz后缀（给起了个名字：标识符）可以替换成我好给的东西
        #在前面生成preprocess文件的时候附带生成一个文件，里面保留文件信息，例如{fold:0,train:[liver_1,liver_2],test:[liver_3]}
        # load the datasets for training and validation. Note that we always draw random samples so we really don't
        # care about distributing training cases across GPUs.
        dataset_tr = nnUNetDataset(self.preprocessed_dataset_folder, tr_keys,
                                   num_images_properties_loading_threshold=0)
        dataset_val = nnUNetDataset(self.preprocessed_dataset_folder, val_keys,
                                    num_images_properties_loading_threshold=0)
        return dataset_tr, dataset_val

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                   initial_patch_size,
                                   self.patch_size,
                                   oversample_foreground_percent=self.oversample_foreground_percent,
                                   sampling_probabilities=None, pad_sides=None, annotated_classes_key=self.annotated_classes_key)
        dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                    self.patch_size,
                                    self.patch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None, annotated_classes_key=self.annotated_classes_key)
        return dl_tr, dl_val
    def get_dataloaders(self):
        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        #deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(patch_size=patch_size, rotation_for_DA=rotation_for_DA, mirror_axes=mirror_axes)

        # validation pipeline
        val_transforms = self.get_validation_transforms()

        #dl_tr, dl_val = self.get_plain_dataloaders(initial_patch_size, dim)
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = nnUNetDataLoader3D(dataset_tr, self.batch_size,
                                   initial_patch_size,
                                   self.patch_size,
                                   oversample_foreground_percent=self.oversample_foreground_percent,
                                   sampling_probabilities=None, pad_sides=None,
                                   annotated_classes_key=self.annotated_classes_key, transforms=tr_transforms)
        dl_val = nnUNetDataLoader3D(dataset_val, self.batch_size,
                                    self.patch_size,
                                    self.patch_size,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    sampling_probabilities=None, pad_sides=None,
                                    annotated_classes_key=self.annotated_classes_key, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        #+++++++++++++++++++++++++++++++++++这里回头要改回来++++++++++++++++++++++++++++++++++++++++++++
        #print(allowed_num_processes)
        #allowed_num_processes=0
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            #核心任务是从数据加载器 (data_loader) 中提取数据项 (item)，然后将其传递给转换操作 (transform) 进行计算，并返回计算后的数据项。
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            '''mt_gen_train = LimitedLenWrapper(self.num_iterations_per_epoch, data_loader=dl_tr, transform=tr_transforms,
                                             num_processes=allowed_num_processes, num_cached=6, seeds=None,
                                             pin_memory=self.device.type == 'cuda', wait_time=0.02)
            mt_gen_val = LimitedLenWrapper(self.num_val_iterations_per_epoch, data_loader=dl_val,
                                           transform=val_transforms, num_processes=max(1, allowed_num_processes // 2),
                                           num_cached=3, seeds=None, pin_memory=self.device.type == 'cuda',
                                           wait_time=0.02)'''
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)

        return mt_gen_train, mt_gen_val


    def on_train_start(self):
        print('成功进入on_train_start')
        self.initialize()
        #判断有没有这个文件夹，没有就生成
        print(self.output_folder)
        self.maybe_mkdir_p(self.output_folder)

        #self.print_plans()
        #清空cache
        self.empty_cache(self.device)
        # dataloaders must be instantiated here because they need access to the training data which may not be present
        # when doing inference
        self.dataloader_train, self.dataloader_val = self.get_dataloaders()

    def on_epoch_start(self):
        self.logger.log('epoch_start_timestamps', time(), self.current_epoch)

    def on_train_epoch_start(self):
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        # lrs are the same for all workers so we don't need to gather them in case of DDP training
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)


    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        semi_infor = batch['semi_infor']
        # 检查是否存在非零元素
        #non_zero_count = torch.nonzero(target).size(0)
        # 输出非零元素的数量
        #print("mask里非零元素的数量:", non_zero_count)
        #print('in Main data.shape',data.shape)

        #data = data.to(self.device, non_blocking=True)
        #semi_infor = semi_infor.to(self.device, non_blocking=True)
        concatenated_data = torch.cat([data, semi_infor], dim=1)
        concatenated_data = concatenated_data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            #print(concatenated_data.shape)
            output = self.network(concatenated_data)
            # del data
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, train_outputs: List[dict]):
        outputs = collate_outputs(train_outputs)

        loss_here = np.mean(outputs['loss'])
       # print(loss_here)
        self.logger.log('train_losses', loss_here, self.current_epoch)

    def on_validation_epoch_start(self):
        self.network.eval()

    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']
        semi_infor = batch['semi_infor']
        # 检查是否存在非零元素
        #non_zero_count = torch.nonzero(target).size(0)
        # 输出非零元素的数量
        #print("mask里非零元素的数量:", non_zero_count)

        #data = data.to(self.device, non_blocking=True)
        #semi_infor = semi_infor.to(self.device, non_blocking=True)
        concatenated_data = torch.cat([data, semi_infor], dim=1)
        concatenated_data = concatenated_data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(concatenated_data)
            del data
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        '''if self.enable_deep_supervision:
            output = output[0]
            target = target[0]'''

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))


        # no need for softmax
        output_seg = output.argmax(1)[:, None]
        predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
        predicted_segmentation_onehot.scatter_(1, output_seg, 1)
        del output_seg

        '''if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:'''

        mask = None
        #predict1 = F.softmax(output, dim=1)
        #predict2 = torch.argmax(predict1, dim=1, keepdim=True)

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=None)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()

        tp_hard = tp_hard[1:]
        fp_hard = fp_hard[1:]
        fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        outputs_collated = collate_outputs(val_outputs)
        tp = np.sum(outputs_collated['tp_hard'], 0)
        fp = np.sum(outputs_collated['fp_hard'], 0)
        fn = np.sum(outputs_collated['fn_hard'], 0)

        loss_here = np.mean(outputs_collated['loss'])

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log('mean_fg_dice', mean_fg_dice, self.current_epoch)
        self.logger.log('dice_per_class_or_region', global_dc_per_class, self.current_epoch)
        self.logger.log('val_losses', loss_here, self.current_epoch)

    def save_checkpoint(self, filename: str) -> None:
        if self.local_rank == 0:
            if not self.disable_checkpointing:

                mod = self.network
                '''if isinstance(mod, OptimizedModule):
                    mod = mod._orig_mod'''

                checkpoint = {
                    'network_weights': mod.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler.state_dict() if self.grad_scaler is not None else None,
                    'logging': self.logger.get_checkpoint(),
                    '_best_ema': self._best_ema,
                    'current_epoch': self.current_epoch + 1,
                    'init_args': self.my_init_kwargs,
                    'trainer_name': self.__class__.__name__,
                    'inference_allowed_mirroring_axes': self.inference_allowed_mirroring_axes,
                }
                torch.save(checkpoint, filename)
            else:
                self.print_to_log_file('No checkpoint written, checkpointing is disabled')

    def on_epoch_end(self):
        self.logger.log('epoch_end_timestamps', time(), self.current_epoch)

        self.print_to_log_file('train_loss', np.round(self.logger.my_fantastic_logging['train_losses'][-1], decimals=4))
        self.print_to_log_file('val_loss', np.round(self.logger.my_fantastic_logging['val_losses'][-1], decimals=4))
        self.print_to_log_file('Pseudo dice', [np.round(i, decimals=4) for i in
                                               self.logger.my_fantastic_logging['dice_per_class_or_region'][-1]])
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], decimals=2)} s")

        # handling periodic checkpointing
        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (self.num_epochs - 1):
            self.save_checkpoint(join(self.output_folder, f'checkpoint_latest_epoch_{current_epoch + 1}.pth'))

        # handle 'best' checkpointing. ema_fg_dice is computed by the logger and can be accessed like this
        if self._best_ema is None or self.logger.my_fantastic_logging['ema_fg_dice'][-1] > self._best_ema:
            self._best_ema = self.logger.my_fantastic_logging['ema_fg_dice'][-1]
            self.print_to_log_file(f"Yayy! New best EMA pseudo Dice: {np.round(self._best_ema, decimals=4)}")
            self.save_checkpoint(join(self.output_folder, 'checkpoint_best.pth'))

        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    def on_train_end(self):
        # dirty hack because on_epoch_end increments the epoch counter and this is executed afterwards.
        # This will lead to the wrong current epoch to be stored
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        # now we can delete latest
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders
        old_stdout = sys.stdout
        with open(os.devnull, 'w') as f:
            sys.stdout = f
            if self.dataloader_train is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_train._finish()
            if self.dataloader_val is not None and \
                    isinstance(self.dataloader_train, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                self.dataloader_val._finish()
            sys.stdout = old_stdout

        self.empty_cache(self.device)
        self.print_to_log_file("Training done.")

    def run_training(self):
        #print('成功进入run training')
        self.on_train_start()

        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                #print(batch_id)
                train_outputs.append(self.train_step(next(self.dataloader_train)))

            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)
            self.on_epoch_end()
        self.on_train_end()

