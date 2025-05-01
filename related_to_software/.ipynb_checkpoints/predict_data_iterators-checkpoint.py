import multiprocessing
import queue
from torch.multiprocessing import Event, Process, Queue, Manager

from time import sleep
from typing import Union, List, Dict, Any
from queue import Empty
import numpy as np
import torch
from related_to_software.simpleitk_reader_writer import SimpleITKIO
from related_to_software.predict_Default_preprocess import DefaultPreprocessor
import matplotlib.pyplot as plt

class Preprocessor_intensity(object):
    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def scale_intensity_ranged(self,image,intensityproperties):
        # print(2)
        # 将输入范围限制在 a_min 和 a_max 之间
        #image = image[0]
        # print(image.shape)
        # print(3)
        '''clipped_image = np.clip(image, a_min, a_max)
        scaled_image = (clipped_image - a_min) * (b_max - b_min) / (a_max - a_min) + b_min'''
        # 如果 clip=True，则将输出范围限制在 0 到 1 之间
        
        '''mean_intensity = 523.47
        std_intensity = 322.51
        lower_bound = 0
        upper_bound = 1627'''
        mean_intensity = intensityproperties['mean']
        std_intensity = intensityproperties['std']
        lower_bound = intensityproperties['percentile_00_5']
        upper_bound = intensityproperties['percentile_99_5']
        
        np.clip(image, lower_bound, upper_bound, out=image)
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image

    def scale_intensity_range(self,img, a_min=-255, a_max=255.0, b_min=None, b_max=None, clip=False, dtype=np.float32):
        if a_max - a_min == 0.0:
            print("Divide by zero (a_min == a_max)")
            if b_min is None:
                return img - a_min
            return img - a_min + b_min

        img = (img - a_min) / (a_max - a_min)

        if b_min is not None and b_max is not None:
            img = img * (b_max - b_min) + b_min

        if clip:
            img = np.clip(img, b_min, b_max)
        return img.astype(dtype)
def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                     list_of_semis: List[np.ndarray],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     intensityproperties: Dict[str, Any],
                                     target_queue: Queue,
                                     done_event: Event,
                                     abort_event: Event,
                                     verbose: bool = False):
    #print(list_of_images)
    preprocessor = Preprocessor_intensity()
    #preprocessor = DefaultPreprocessor(verbose=verbose)

    try:
        for idx in range(len(list_of_images[0])):
            #print(idx, len(list_of_images[0]))
            data = preprocessor.scale_intensity_ranged(list_of_images[0][idx],intensityproperties)

            semi = list_of_semis[0][idx]
            data_combined = np.concatenate((data, semi), axis=0)

            data_transposed = torch.from_numpy(data_combined).contiguous().float()
            item = {'data': data_transposed, 'data_properties': list_of_image_properties[0][idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e


def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                   list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                   list_of_semis: List[np.ndarray],
                                   list_of_image_properties: List[dict],
                                   truncated_ofnames: Union[List[str], None],
                                   intensityproperties:Dict[str, Any],
                                   num_processes: int,
                                   verbose: bool = False,
                                   pin_memory: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_images), num_processes)
    assert num_processes >= 1
    target_queues = []
    processes = []
    done_events = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = manager.Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromnpy_save_to_queue,
                     args=(
                         list_of_images[i::num_processes],
                         list_of_segs_from_prev_stage[i::num_processes] if list_of_segs_from_prev_stage is not None else None,
                         list_of_semis[i::num_processes] if list_of_semis is not None else None,
                         list_of_image_properties[i::num_processes],
                         truncated_ofnames[i::num_processes] if truncated_ofnames is not None else None,
                         intensityproperties,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        done_events.append(event)
        processes.append(pr)
        target_queues.append(queue)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]


def read_images_from_paths(image_path,semi_path):

    simple_itk_io = SimpleITKIO()

    img, prop = simple_itk_io.read_images([image_path])
    semi, prop_no_use = simple_itk_io.read_images([semi_path])

    return img, semi, prop
def preprocess_fromfiles_save_to_queue(list_of_lists: List[List[str]],
                                       list_of_semis: List[List[str]],
                                       truncated_ofnames: Union[List[str], None],
                                       intensityproperties: Dict[str, Any],
                                       target_queue: Queue,
                                       done_event: Event,
                                       abort_event: Event,
                                       verbose: bool = False):
    try:
        preprocessor = Preprocessor_intensity()
        #print(list_of_lists,truncated_ofnames)
        for idx in range(len(list_of_lists[0])):
            #print(idx,len(list_of_lists[0]))
            img, semi, prop = read_images_from_paths(list_of_lists[0][idx], list_of_semis[0][idx])
            #print("Shape of the first image:",img.shape)
            #print("Shape of the first image:",img[0].shape)
            data = preprocessor.scale_intensity_ranged(img,intensityproperties)

            data_combined = np.concatenate((data, semi), axis=0)
            #print(f'data:{data_combined.shape}')
            data_combined = np.array(data_combined)
            data_transposed = torch.from_numpy(data_combined).contiguous().float()

            item = {'data': data_transposed, 'data_properties': prop,
                    'ofile': truncated_ofnames[0][idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    if abort_event.is_set():
                        return
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    pass
        done_event.set()
    except Exception as e:
        # print(Exception, e)
        abort_event.set()
        raise e

def preprocessing_iterator_fromfiles(list_of_lists: List[List[str]],
                                     list_of_semis: List[List[str]],
                                     truncated_ofnames: List[List[str]],
                                     intensityproperties:Dict[str, Any],
                                     num_processes: int,
                                     pin_memory: bool = False,
                                     verbose: bool = False):
    context = multiprocessing.get_context('spawn')
    manager = Manager()
    num_processes = min(len(list_of_lists), num_processes)
    assert num_processes >= 1
    processes = []
    done_events = []
    target_queues = []
    abort_event = manager.Event()
    for i in range(num_processes):
        event = manager.Event()
        queue = Manager().Queue(maxsize=1)
        pr = context.Process(target=preprocess_fromfiles_save_to_queue,
                     args=(
                         list_of_lists[i::num_processes],
                         list_of_semis[i::num_processes],
                         truncated_ofnames[i::num_processes],
                         intensityproperties,
                         queue,
                         event,
                         abort_event,
                         verbose
                     ), daemon=True)
        pr.start()
        target_queues.append(queue)
        done_events.append(event)
        processes.append(pr)

    worker_ctr = 0
    while (not done_events[worker_ctr].is_set()) or (not target_queues[worker_ctr].empty()):
        # import IPython;IPython.embed()
        if not target_queues[worker_ctr].empty():
            item = target_queues[worker_ctr].get()
            worker_ctr = (worker_ctr + 1) % num_processes
        else:
            all_ok = all(
                [i.is_alive() or j.is_set() for i, j in zip(processes, done_events)]) and not abort_event.is_set()
            if not all_ok:
                raise RuntimeError('Background workers died. Look for the error message further up! If there is '
                                   'none then your RAM was full and the worker was killed by the OS. Use fewer '
                                   'workers or get more RAM in that case!')
            sleep(0.01)
            continue
        if pin_memory:
            [i.pin_memory() for i in item.values() if isinstance(i, torch.Tensor)]
        yield item
    [p.join() for p in processes]
#可以跑的单线程版本

'''def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     dataset_json: dict,
                                     target_queue: Queue,
                                     verbose: bool = False):
    try:
        for idx in range(len(list_of_images)):
            data = scale_intensity_ranged(list_of_images[idx])
            data = np.expand_dims(data, axis=0)
            data = torch.from_numpy(data).contiguous().float()

            item = {'data': data, 'data_properties': list_of_image_properties[idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            target_queue.put(item)
    except Exception as e:
        raise e

def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                        list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                        list_of_image_properties: List[dict],
                                        truncated_ofnames: Union[List[str], None],
                                        batch_size: int,
                                        dataset_json: dict,
                                        pin_memory: bool = False,
                                        verbose: bool = False):
    target_queue = Queue()

    preprocess_fromnpy_save_to_queue(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_ofnames, dataset_json, target_queue, verbose)

    while True:
        try:
            item = target_queue.get(timeout=0.1)
            yield item
        except queue.Empty:
            break

    return'''

'''def preprocess_fromnpy_save_to_queue(list_of_images: List[np.ndarray],
                                     list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                     list_of_image_properties: List[dict],
                                     truncated_ofnames: Union[List[str], None],
                                     target_queue: Queue,
                                     done_event: Event,
                                     abort_event: Event,
                                     verbose: bool = False):
    try:
        for idx in range(len(list_of_images)):
            if abort_event.is_set():
                return
            data = scale_intensity_ranged(list_of_images[idx])
            data = np.expand_dims(data, axis=0)
            data = torch.from_numpy(data).contiguous().float()

            item = {'data': data, 'data_properties': list_of_image_properties[idx],
                    'ofile': truncated_ofnames[idx] if truncated_ofnames is not None else None}
            success = False
            while not success:
                try:
                    target_queue.put(item, timeout=0.01)
                    success = True
                except queue.Full:
                    if abort_event.is_set():
                        return
                    pass
        done_event.set()
    except Exception as e:
        abort_event.set()
        raise e




def preprocessing_iterator_fromnpy(list_of_images: List[np.ndarray],
                                   list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                                   list_of_image_properties: List[dict],
                                   truncated_ofnames: Union[List[str], None],
                                   num_processes: int,
                                   batch_size: int=1,
                                   verbose: bool = False):
    processes = []
    done_events = []
    abort_event = multiprocessing.Event()
    queues = [multiprocessing.Queue(maxsize=batch_size) for _ in range(num_processes)]

    for i in range(num_processes):
        event = multiprocessing.Event()
        pr = multiprocessing.Process(target=preprocess_fromnpy_save_to_queue,
                                     args=(list_of_images[i::num_processes],
                                           list_of_segs_from_prev_stage[
                                           i::num_processes] if list_of_segs_from_prev_stage is not None else None,
                                           list_of_image_properties[i::num_processes],
                                           truncated_ofnames[
                                           i::num_processes] if truncated_ofnames is not None else None,
                                           queues[i], event, abort_event, verbose)
                                     )
        pr.start()
        done_events.append(event)
        processes.append(pr)

    try:
        while True:
            for queue in queues:
                try:
                    item = queue.get(timeout=0.1)
                    yield item
                except queue.Empty:  # Change 'queue.Empty' to 'multiprocessing.queues.Empty'
                    pass

            if all(event.is_set() for event in done_events) or abort_event.is_set():
                break

    except Exception as e:
        abort_event.set()
        raise e

    for process in processes:
        process.join()'''

