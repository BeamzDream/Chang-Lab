import numpy as np 
import spikeinterface.extractors as se
import spikeinterface.sorters as ss

from mountainsort4_1_0 import sort_dataset as ms4_sort_dataset # MountainSort spike sorting
from validate_sorting_results import validate_sorting_results # Validation processors

import time

start_time = time.time()

# acc_1_04202016_kurosawacoppola_pre.mda
recording2 = se.MdaRecordingExtractor(folder_path='raw_mda_files')
print('Num. channels = {}'.format(len(recording2.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording2.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording2.get_num_frames()))

print("Extraction took", time.time() - start_time, "to run")

start_time = time.time()

default_ms4_params = ss.Mountainsort4Sorter.default_params()
print(default_ms4_params)

output_dir = 'output_folder'
dsdir = 'raw_mda_files'
# Mountainsort4 spike sorting
ms4_sort_dataset(dataset_dir=dsdir,output_dir=output_dir,adjacency_radius=-1,detect_threshold=3)
# A=validate_sorting_results(dataset_dir=dsdir,sorting_output_dir=output_dir,output_dir=output_dir)
# amplitudes_true=A['amplitudes_true']
# accuracies=A['accuracies']

# sorting_MS4 = ss.run_mountainsort4(recording=recording2, **default_ms4_params,
#                                    output_dir='tmp_MS4')

print("Sorting took", time.time() - start_time, "to run")


