import numpy as np 
import spikeinterface.extractors as se
import spikeinterface.sorters as ss

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

# Mountainsort4 spike sorting
sorting_MS4 = ss.run_mountainsort4(recording=recording2, **default_ms4_params,
                                   output_folder='tmp_MS4')

print("Sorting took", time.time() - start_time, "to run")


