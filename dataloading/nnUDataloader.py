import numpy as np
from dataloading.nnUbase_loader import nnUNetDataLoaderBase
from dataloading.nnUDataset import nnUNetDataset
import torch
from threadpoolctl import threadpool_limits
class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        #print('in uundataloader,selected_keys:',selected_keys)
        # preallocate memory for data and seg
        #print('self.data_shape:',self.data_shape)
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        semi_infor_all = np.zeros(self.semi_infor_shape, dtype=np.float32)
        case_properties = []

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, semi_infor, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            #shape = data.shape[1:]
            shape = data.shape
            #print('in uundataloader,shape: ',shape)
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])
            #print(f'bbox_lbs:{bbox_lbs}, bbox_ubs:{bbox_ubs}')
            #print('in nnudataloader,bbox_lbs, bbox_ubs',bbox_lbs, bbox_ubs)
            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
            valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]
            #print(f'valid_bbox_lbs:{valid_bbox_lbs}, valid_bbox_ubs:{valid_bbox_ubs}')
            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            #print("data shape:", data.shape)
            #print(valid_bbox_lbs)
            #print("Length of valid_bbox_lbs:", len(valid_bbox_lbs))
            #print("Length of valid_bbox_ubs:", len(valid_bbox_ubs))
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)][1:])
            #this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            #print(this_slice)
            data = data[this_slice]

            this_slice = tuple([slice(0, semi_infor.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)][1:])
            semi_infor = semi_infor[this_slice]

            #this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)][1:])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            #print('padding ',padding)
            #print('data.shape ', data.shape)
            #data_all[j] = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
            #seg_all[j] = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=0)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            #print('in nnudataloader, data_all[j].shape',data_all[j].shape)

            semi_infor_all[j] = np.pad(semi_infor, padding, 'constant', constant_values=0)
            #print('in nnudataloader, semi_infor_all[j].shape', semi_infor_all[j].shape)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=0)
        if self.transforms is not None:
            if torch is not None:
                torch_nthreads = torch.get_num_threads()
                torch.set_num_threads(1)
            with threadpool_limits(limits=1, user_api=None):
                data_all = torch.from_numpy(data_all).float()
                semi_infor_all = torch.from_numpy(semi_infor_all).float()
                seg_all = torch.from_numpy(seg_all).to(torch.int16)
                images = []
                semi_infors = []
                segs = []

                for b in range(self.batch_size):
                    #print(b,data_all[b].shape)
                    #np.expand_dims(s, axis=0)
                    tmp = self.transforms(**{'data': np.expand_dims(data_all[b], axis=0), 'semi_infor':np.expand_dims(semi_infor_all[b], axis=0) ,'seg': np.expand_dims(seg_all[b], axis=0)})
                    images.append(tmp['data'].squeeze(axis=0))
                    semi_infors.append(tmp['semi_infor'].squeeze(axis=0))
                    segs.append(tmp['seg'].squeeze(axis=0))
                data_all = torch.stack(images)
                semi_infor_all = torch.stack(semi_infors)
                #print(data_all.shape)
                #seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                seg_all = torch.stack(segs)
                del segs, images, semi_infors
            if torch is not None:
                torch.set_num_threads(torch_nthreads)
            #print('data_all',data_all)
            return {'data': data_all, 'target': seg_all, 'semi_infor':semi_infor_all, 'properties': case_properties,'keys': selected_keys}

        return {'data': data_all, 'seg': seg_all, 'semi_infor':semi_infor_all,'properties': case_properties, 'keys': selected_keys}


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)  # this should not load the properties!
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
