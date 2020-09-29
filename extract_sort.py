import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import os
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import scipy.io
import h5py
import hdf5storage
import time

def extract_recording(timeseries):
  start_time = time.time()
  sampling_frequency = 40000
  geom = np.zeros((num_channels, 2))
  geom[:, 0] = range(num_channels)
  recording = se.NumpyRecordingExtractor(timeseries=timeseries, geom=geom, sampling_frequency=sampling_frequency)
  print('Num. channels = {}'.format(len(recording.get_channel_ids())))
  print('Sampling frequency = {} Hz'.format(recording.get_sampling_frequency()))
  print('Num. timepoints = {}'.format(recording.get_num_frames()))

  print("Extraction time:", time.time() - start_time)

  return recording

def preprocess_recording(recording, freq_min, freq_max):
  """ Bandpass filter for recording """

  start_time = time.time()
  recording_f = st.preprocessing.bandpass_filter(recording, freq_min=freq_min, freq_max=freq_max)
  print("Pre-processing time:", time.time() - start_time)

  return recording_f


def sort_recording(recording, file_name):
  
  start_time = time.time()

  output_dir = '../tmp_MS4/'
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  default_ms4_params = ss.Mountainsort4Sorter.default_params()
  default_ms4_params['detect_threshold'] = 4
  default_ms4_params['curation'] = False
  sorting = ss.run_mountainsort4(recording=recording, **default_ms4_params, output_folder=output_dir+file_name)

  print("Sorting time:", time.time() - start_time)

  return sorting

def postprocess_recording(recording, sorting, file_name):
  start_time = time.time()

  output_dir = '../average_waveform/'
  if not os.path.exists(output_dir):
    os.mkdir(output_dir)

  wf = st.postprocessing.get_unit_waveforms(recording, sorting, ms_before=1, ms_after=2,
                                            save_as_features=True, verbose=True)
  print("Shared unit spike feature names: ", sorting.get_shared_unit_spike_feature_names())
  print("Waveforms (n_spikes, n_channels, n_points)", wf[0].shape)

  max_chan = st.postprocessing.get_unit_max_channels(recording, sorting, save_as_property=True, verbose=True)
  savename = output_dir+file_name+'_avg_waveforms.mat'
  scipy.io.savemat(savename,{'wf':wf,'maxchn':max_chan}, do_compression=True)

  templates = st.postprocessing.get_unit_templates(recording, sorting, max_spikes_per_unit=200,
                                                  save_as_property=True, verbose=True)

  print("Post-processing time: ", time.time() - start_time)

  return wf, max_chan, templates

if __name__ == "__main__":
  directory_in_str = "../raw_mat_files"
  directory = os.fsencode(directory_in_str)
  for file in os.listdir(directory):
    file_name = os.fsdecode(file)
    file_path = os.path.join(directory_in_str, file_name)
    file_name = file_name.replace('.mat', '')
    f = h5py.File(file_path, 'r')
    num_channels = f['mat'].shape[1]
    ts = np.transpose(np.array(f['mat']))
    f.close()

    recording = extract_recording(timeseries=ts)
    recording_f = preprocess_recording(recording=recording, freq_min=300, freq_max=6000)
    sorting = sort_recording(recording=recording_f, file_name=file_name)
    wf, max_chan, templates = postprocess_recording(recording = recording, sorting = sorting, file_name = file_name)

    output_dir = "waveform_visualization/"
    if not os.path.exists(output_dir):
      os.mkdir(output_dir)

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    for i in range(len(wf)):
      axs[0].plot(wf[i][:, 0, :].T, lw=0.3)
      axs[1].plot(templates[i].T)
    axs[0].set_title('Waveform visualization for '+ file_name)
    axs[1].set_title('Template visualization for '+ file_name)
    fig.savefig(output_dir + file_name + '_waveform.png')
