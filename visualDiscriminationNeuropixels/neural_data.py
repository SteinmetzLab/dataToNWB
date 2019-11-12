from datetime import datetime
from dateutil.tz import tzlocal
import pandas as pd
import numpy as np
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
import pynwb
from pynwb import NWBHDF5IO, NWBFile

nwb_file = NWBFile(
    session_description='Test to see if building a file works',
    identifier='Test123',
    session_start_time=datetime(2016, 12, 14, tzinfo=tzlocal()),
    file_create_date=datetime.now(tzlocal()),
    experimenter='name',
    experiment_description='description',
    institution='institution'  # add/fill out for real file
)


def read_npy_file(filename):
    """
    Loads .npy file into numpy array
    :param filename: name of the file being loaded, .npy file
    :return: numpy array
    """
    np_arr = np.load(filename)
    return np_arr


# Devices and Electrode Groups
probe_descriptions = pd.read_csv('probes.description.tsv', sep='\t')
probe_descriptions = list(probe_descriptions['description'])
electrode_groups = list()
for i in range(len(probe_descriptions)):
    probe_device = Device(name=str(i))
    probe_electrode_group = ElectrodeGroup(
        name=str(i),
        description='Neuropixels Phase3A opt3',
        device=probe_device,
        location=''
        # sampling_rate=30000.0,
        # lfp_sampling_rate=2500.0,
        # has_lfp_data=True,
    )
    nwb_file.add_device(probe_device)
    electrode_groups.append(probe_electrode_group)
    nwb_file.add_electrode_group(probe_electrode_group)
insertion_df = pd.read_csv('probes.insertion.tsv', sep='\t')
insertion_df['probes'] = insertion_df.index.values

# Channel Table
channel_site = read_npy_file('channels.site.npy')
channel_brain = pd.read_csv('channels.brainLocation.tsv', sep='\t')
channel_probes = read_npy_file('channels.probe.npy')
channel_probes = np.ravel(channel_probes.astype(int))
channel_site_pos = read_npy_file('channels.sitePositions.npy')
channel_table = pd.DataFrame(columns=['probes'])
channel_table['probes'] = channel_probes
channel_table = channel_table.merge(insertion_df, 'left', 'probes')
channel_table = pd.concat([channel_table, channel_brain], axis=1)
nwb_file.electrodes = pynwb.file.ElectrodeTable().from_dataframe(channel_table, name='electrodes')
nwb_file.add_electrode_column(
    name='site_id',
    description='The site number, in within-probe numbering, of the channel '
                '(in practice for this dataset this always starts at zero and '
                'counts up to 383 on each probe so is equivalent to the channel '
                'number - but if switches had been used, the site number could '
                'have been different than the channel number).',
    data=np.ravel(channel_site)
)
nwb_file.add_electrode_column(
    name='site_position',
    description='The x- and y-position of the site relative to the face of the probe '
                '(where the first column is across the face of the probe laterally '
                'and the second is the position along the length of the probe; '
                'the sites nearest the tip have second column=0).',
    data=channel_site_pos
)
nwb_file.add_electrode_column(
    name='group',
    description='electrode group for each channel',
    data=[electrode_groups[c] for c in channel_probes]
)

# Clusters/Spikes
clusters = read_npy_file('clusters.probes.npy')
cluster_probe = read_npy_file('clusters.probes.npy')
cluster_probe = np.ravel(cluster_probe.astype(int))
cluster_channel = read_npy_file('clusters.peakChannel.npy')
phy_annotations = np.ravel(read_npy_file('clusters._phy_annotation.npy'))
waveform_chans = read_npy_file('clusters.templateWaveformChans.npy')
waveform_chans = waveform_chans.astype(int)
waveform = read_npy_file('clusters.templateWaveforms.npy')
waveform_duration = read_npy_file('clusters.waveformDuration.npy')
spike_to_clusters = read_npy_file('spikes.clusters.npy')
spike_times = read_npy_file('spikes.times.npy')
spike_amps = read_npy_file('spikes.amps.npy')
spike_amps = np.ravel(spike_amps)
spike_depths = read_npy_file('spikes.depths.npy')

# TODO: Take out index bounds later
# Sorting spikes into clusters
spike_to_clusters = spike_to_clusters[:200, :]
cluster_info = dict()
for i in range(len(spike_to_clusters)):
    s = int(spike_to_clusters[i])
    if s not in cluster_info:
        cluster_info[s] = [i]
    else:
        cluster_info[s].append(i)

# Can take out after index bounds are removed when all clusters are included
depths = list()
annotations = list()
c_channel = list()
wave_duration = list()

# Add Units
for i in cluster_info:
    c = cluster_info[i]
    times = np.array(spike_times[c])
    interval = np.empty((1, 2))
    interval[0, 0] = times.min()
    interval[0, 1] = times.max()

    nwb_file.add_unit(
        spike_times=np.ravel(times),
        obs_intervals=interval,
        electrodes=np.ravel(waveform_chans[c, :]),
        electrode_group=electrode_groups[cluster_probe[i]],
        # waveform_mean=waveform[c, :, :],
        id=i
    )
    # take out once bounds are removed
    depths.append(np.array(np.ravel(spike_depths[c])))
    annotations.append(phy_annotations[c])
    c_channel.append(cluster_channel[c])
    wave_duration.append(cluster_channel[c])


amps = np.array([spike_amps[cluster_info[i]] for i in cluster_info.keys()])
# amps = np.array(amps)
depths = np.array(depths)
c_channel = np.array(c_channel)
wave_duration = np.array(wave_duration)


# Add unit columns
"""
nwb_file.add_unit_column(
    name='spike_amps',
    description='The peak-to-trough amplitude, obtained from the template and '
                'template-scaling amplitude returned by Kilosort (not from the raw data).',
    data=amps
)
nwb_file.add_unit_column(
    name='spike_depths',
    description='The position of the center of mass of the spike on the probe, '
                'determined from the principal component features returned by Kilosort. '
                'The deepest channel on the probe is depth=0, and the most superficial is depth=3820.',
    data=depths
)
nwb_file.add_unit_column(
    name='phy_annotations',
    description='0 = noise (these are already excluded and don\'t appear in this '
                'dataset at all); 1 = MUA (i.e. presumed to contain spikes from multiple '
                'neurons; these are not analyzed in any analyses in the paper); 2 = Good '
                '(manually labeled); 3 = Unsorted. In this dataset \'Good\' was applied '
                'in a few but not all datasets to included neurons, so in general the '
                'neurons with _phy_annotation>=2 are the ones that should be included.',
    data=annotations
)
nwb_file.add_unit_column(
    name='peak_channel',
    description='The channel number of the location of the peak of the cluster\'s waveform.',
    data=c_channel
)
nwb_file.add_unit_column(
    name='waveform_duration',
    description='The trough-to-peak duration of the waveform on the peak channel.',
    data=wave_duration
)
"""
with NWBHDF5IO('test_neural_nwb_file.nwb', 'w') as io:
    io.write(nwb_file)
    print('saved')
