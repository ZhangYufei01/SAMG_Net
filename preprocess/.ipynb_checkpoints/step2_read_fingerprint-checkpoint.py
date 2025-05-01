import json
from related_to_software.simpleitk_reader_writer import SimpleITKIO
import multiprocessing
import numpy as np
from typing import List, Type, Union
from tqdm import tqdm
from time import sleep
import os



def collect_foreground_intensities(segmentation: np.ndarray, images: np.ndarray, seed: int = 1234,
                                       num_samples: int = 10000):
    """
    images=image with multiple channels = shape (c, x, y(, z))
    """
    assert images.ndim == 4 and segmentation.ndim == 4
    assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("
    assert not np.any(np.isnan(images)), "Images contains NaN values. grrrr.... :-("

    rs = np.random.RandomState(seed)

    intensities_per_channel = []
    # we don't use the intensity_statistics_per_channel at all, it's just something that might be nice to have
    intensity_statistics_per_channel = []

    # segmentation is 4d: 1,x,y,z. We need to remove the empty dimension for the following code to work
    foreground_mask = segmentation[0] > 0
    percentiles = np.array((0.5, 50.0, 99.5))

    for i in range(len(images)):
        foreground_pixels = images[i][foreground_mask]
        num_fg = len(foreground_pixels)
        # sample with replacement so that we don't get issues with cases that have less than num_samples
        # foreground_pixels. We could also just sample less in those cases but that would than cause these
        # training cases to be underrepresented
        intensities_per_channel.append(
            rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else [])

        mean, median, mini, maxi, percentile_99_5, percentile_00_5 = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        if num_fg > 0:
            percentile_00_5, median, percentile_99_5 = np.percentile(foreground_pixels, percentiles)
            mean = np.mean(foreground_pixels)
            mini = np.min(foreground_pixels)
            maxi = np.max(foreground_pixels)

        intensity_statistics_per_channel.append({
            'mean': mean,
            'median': median,
            'min': mini,
            'max': maxi,
            'percentile_99_5': percentile_99_5,
            'percentile_00_5': percentile_00_5,

        })

    return intensities_per_channel, intensity_statistics_per_channel

def analyze_case(image_files: List[str], segmentation_file: str, num_samples: int = 10000):
    rw = SimpleITKIO()
    #print(image_files)
    images, properties_images = rw.read_images([image_files])
    segmentation, properties_seg = rw.read_seg([segmentation_file])

    # we no longer crop and save the cropped images before this is run. Instead we run the cropping on the fly.
    # Downside is that we need to do this twice (once here and once during preprocessing). Upside is that we don't
    # need to save the cropped data anymore. Given that cropping is not too expensive it makes sense to do it this
    # way. This is only possible because we are now using our new input/output interface.
    #data_cropped, seg_cropped, bbox = crop_to_nonzero(images, segmentation)

    foreground_intensities_per_channel, foreground_intensity_stats_per_channel = collect_foreground_intensities(segmentation, images, num_samples=num_samples)
    #print(foreground_intensities_per_channel)
    return foreground_intensities_per_channel, foreground_intensity_stats_per_channel


# 读取 JSON 文件
def process_dataset(json_file_path: str, num_workers: int = 4):
    with open(json_file_path, 'r') as file:
        config = json.load(file)
    dataset = config['data']
    num_foreground_samples_per_case = int(10e7 // len(dataset))
    r = []
    with multiprocessing.Pool(num_workers) as p:
        for data in dataset:
            r.append(p.starmap_async(analyze_case, ((data['image'], data['label'], num_foreground_samples_per_case),)))
        remaining = list(range(len(dataset)))
        workers = [j for j in p._pool]
        with tqdm(desc=None, total=len(dataset)) as pbar:
            while len(remaining) > 0:
                all_alive = all([j.is_alive() for j in workers])
                if not all_alive:
                    raise RuntimeError('Some background worker is 6 feet under. Yuck. \n'
                                       'OK jokes aside.\n'
                                       'One of your background processes is missing. This could be because of '
                                       'an error (look for an error message) or because it was killed '
                                       'by your OS due to running out of RAM. If you don\'t see '
                                       'an error message, out of RAM is likely the problem. In that case '
                                       'reducing the number of workers might help')
                done = [i for i in remaining if r[i].ready()]
                for _ in done:
                    pbar.update()
                remaining = [i for i in remaining if i not in done]
                sleep(0.1)
    results = [i.get()[0] for i in r]
    foreground_intensities_per_channel = [np.concatenate([r[0][i] for r in results]) for i in
                                          range(len(results[0][0]))]
    foreground_intensities_per_channel = np.array(foreground_intensities_per_channel)
    intensity_statistics_per_channel = {}
    percentiles = np.array((0.5, 50.0, 99.5))
    for i in range(len(foreground_intensities_per_channel)):
        percentile_00_5, median, percentile_99_5 = np.percentile(foreground_intensities_per_channel[i],
                                                                 percentiles)
        intensity_statistics_per_channel[i] = {
            'mean': float(np.mean(foreground_intensities_per_channel[i])),
            'median': float(median),
            'std': float(np.std(foreground_intensities_per_channel[i])),
            'min': float(np.min(foreground_intensities_per_channel[i])),
            'max': float(np.max(foreground_intensities_per_channel[i])),
            'percentile_99_5': float(percentile_99_5),
            'percentile_00_5': float(percentile_00_5),
        }
    config['intensity_statistics_per_channel'] = intensity_statistics_per_channel[0]
    with open(json_file_path, 'w') as file:
        json.dump(config, file, indent=4)
    print("JSON update")

#json_file_path = '/dssg/home/acct-clsyzs/clsyzs-zhyf/3DU_NNU_semi_new/experiment/Beigene_liver/data/data_summary.json'
#process_dataset(json_file_path)