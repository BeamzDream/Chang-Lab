import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np 
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.toolkit as st
import scipy.io
import h5py
import hdf5storage
import time

start_time = time.time()

f = h5py.File('../raw_mat_files/acc_1_04052016_kurocoppola_pre.mat', 'r')
num_channels = f['mat'].shape[1]
ts = np.transpose(np.array(f['mat']))
f.close()

sampling_frequency = 40000  # in Hz
geom = np.zeros((num_channels, 2))
geom[:, 0] = range(num_channels)
recording = se.NumpyRecordingExtractor(timeseries=ts, geom=geom, sampling_frequency=sampling_frequency)
print('Num. channels = {}'.format(len(recording.get_channel_ids())))
print('Sampling frequency = {} Hz'.format(recording.get_sampling_frequency()))
print('Num. timepoints = {}'.format(recording.get_num_frames()))

print("Extraction time:", time.time() - start_time)

start_time = time.time()

recording_f = st.preprocessing.bandpass_filter(recording, freq_min=300, freq_max=6000)

print("Pre-processing time:", time.time() - start_time)

start_time = time.time()
default_ms4_params = ss.Mountainsort4Sorter.default_params()
default_ms4_params['detect_threshold'] = 4
default_ms4_params['curation'] = False
file_name = 'acc_1_04072016_kurosawacoppola_pre'
sorting = ss.run_mountainsort4(recording=recording_f, **default_ms4_params, output_folder='../tmp_MS4/'+file_name)

print("Sorting time:", time.time() - start_time)

start_time = time.time()
wf = st.postprocessing.get_unit_waveforms(recording, sorting, ms_before=1, ms_after=2,
                                          save_as_features=True, verbose=True)
print("Shared unit spike feature names: ", sorting.get_shared_unit_spike_feature_names())
print("Waveforms (n_spikes, n_channels, n_points)", wf[0].shape)

fig, ax = plt.subplots()
ax.plot(wf[0][:, 0, :].T, color='k', lw=0.3)
ax.plot(wf[1][:, 0, :].T, color='r', lw=0.3)
ax.plot(wf[2][:, 0, :].T, color='b', lw=0.3)
fig.savefig('waveform_visualization/' + file_name + '_waveform.png')

max_chan = st.postprocessing.get_unit_max_channels(recording, sorting, save_as_property=True, verbose=True)
savename = '../average_waveform/'+file_name+'_avg_waveforms.mat'
scipy.io.savemat(savename,{'wf':wf,'maxchn':max_chan}, do_compression=True)

templates = st.postprocessing.get_unit_templates(recording, sorting, max_spikes_per_unit=200,
                                                 save_as_property=True, verbose=True)

fig, ax = plt.subplots()
ax.plot(templates[0].T, color='k')
ax.plot(templates[1].T, color='r')
ax.plot(templates[2].T, color='b')
ax.plot(templates[3].T, color='y')
fig.savefig('template_visualization/' + file_name + '_template.png')

print("Postprocessing time: ", time.time() - start_time)




