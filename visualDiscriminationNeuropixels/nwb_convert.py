import pynwb
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.device import Device
from pynwb.epoch import TimeIntervals
from pynwb.ecephys import ElectrodeGroup
from pynwb.behavior import BehavioralEvents, BehavioralEpochs, BehavioralTimeSeries, PupilTracking, IntervalSeries
from allensdk.brain_observatory.ecephys.write_nwb.__main__ import add_ragged_data_to_dynamic_table

################################################################################
# CREATE FILE
subject = pynwb.file.Subject(
    age='77',
    genotype='tetO-G6s x CaMK-tTA',
    sex='F',
    species='Mus musculus',
    description='strain: C57Bl6/J'
)
nwb_file = NWBFile(
    session_description='Neuropixels recording during visual discrimination in awake mice.',
    identifier='Cori_2016-12-14',
    session_start_time=datetime(2016, 12, 14, 12, 0, 0, tzinfo=timezone(timedelta(0))),
    file_create_date=datetime.now(tzlocal()),
    institution='University College London',
    lab='The Carandini & Harris Lab',
    subject=subject,
    experimenter='Nick Steinmetz',
    experiment_description='Large-scale Neuropixels recordings across brain regions '
                           'of mice during a head-fixed visual discrimination task. ',
    related_publications='DOI 10.1038/s41586-019-1787-x',
    keywords=['Neural coding', 'Neuropixels', 'mouse', 'brain-wide',
              'vision', 'visual discrimination', 'electrophysiology'],

)

behavior_module = ProcessingModule('behavior', 'behavior module')
nwb_file.add_processing_module(behavior_module)
################################################################################
# PROCESS DATA


def read_npy_file(filename):
    """
    Loads .npy file into numpy array
    :param filename: name of the file being loaded, .npy file
    :return: numpy array
    """
    np_arr = np.load(filename)
    return np_arr


def get_rate(timestamps):
    """
    Gets constant rate for (2, 2) array
    :param timestamps: (2, 2) numpy array
    :return: rate as float
    """
    return (timestamps[1, 1] - timestamps[0, 1]) / (timestamps[1, 0])


################################################################################
# EYE


def eye():
    """
    Adds eye Position and PupilTracking to behavior processing module.
    Needs data from eye.timestamps.npy, eye.area.npy, and eye.xyPos.npy.
    """
    eye_timestamps = read_npy_file('eye.timestamps.npy')[:, 1]  # takes out index
    eye_area = read_npy_file('eye.area.npy')
    eye_xy_pos = read_npy_file('eye.xyPos.npy')
    pupil = TimeSeries(
        name='eye_area',
        timestamps=eye_timestamps,
        data=np.ravel(eye_area),
        unit='arb. unit',
        description='Features extracted from the video of the right eye.',
        comments='The area of the pupil extracted with DeepLabCut. Note that '
                 'it is relatively very small during the discrimination task '
                 'and during the passive replay because the three screens are '
                 'medium-grey at this time and black elsewhere - so the much '
                 'brighter overall luminance levels lead to relatively '
                 'constricted pupils.'
    )
    eye_xy = TimeSeries(
        name='eye_xy_positions',
        timestamps=eye_timestamps,
        data=eye_xy_pos,  # currently as [x, y] pairs
        unit='arb. unit',
        description='Features extracted from the video of the right eye.',
        comments='The 2D position of the center of the pupil in the video '
                 'frame. This is not registered to degrees visual angle, but '
                 'could be used to detect saccades or other changes in eye position.'
    )
    pupil_track = PupilTracking(pupil)
    pupil_track.add_timeseries(eye_xy)
    behavior_module.add_data_interface(pupil_track)


eye()
################################################################################
# FACE


def face_nwb():
    """
    Adds Face Energy BehavioralTimeSeries to behavior processing module.
    Needs data from face.motionEnergy.npy and face.timestamps.npy.
    """
    face_motion_energy = read_npy_file('face.motionEnergy.npy')
    face_timestamps = read_npy_file('face.timestamps.npy')
    face_rate = get_rate(face_timestamps)
    face_energy = TimeSeries(
        name='face_motion_energy',
        data=np.ravel(face_motion_energy),
        unit='arb. unit',
        starting_time=face_timestamps[0, 1],
        rate=face_rate,
        description='Features extracted from the video of the frontal aspect of '
                    'the subject, including the subject\'s face and forearms.',
        comments='The integrated motion energy across the whole frame, i.e. '
                 'sum( (thisFrame-lastFrame)^2 ). Some smoothing is applied '
                 'before this operation.'
    )
    face_interface = BehavioralTimeSeries(face_energy)
    behavior_module.add_data_interface(face_interface)


face_nwb()
################################################################################
# LICK_PIEZO/LICKS


def lick_piezo():
    """
    Adds lick_piezo to acquisition.
    Needs data from lickPiezo.raw.npy and lickPiezo.timestamps.npy.
    """
    lp_raw = read_npy_file('lickPiezo.raw.npy')
    lp_timestamps = read_npy_file('lickPiezo.timestamps.npy')
    lp_rate = get_rate(lp_timestamps)
    lick_piezo_ts = TimeSeries(
        name='lickPiezo',
        starting_time=lp_timestamps[0, 1],
        rate=lp_rate,
        data=np.ravel(lp_raw),
        unit='V',
        description='Voltage values from a thin-film piezo connected to the '
                    'lick spout, so that values are proportional to deflection '
                    'of the spout and licks can be detected as peaks of the signal.'
    )
    nwb_file.add_acquisition(lick_piezo_ts)


lick_piezo()


def lick_times():
    """
    Adds Lick BehavioralEvents to behavior processing module.
    Needs data from licks.times.npy.
    """
    lick_timestamps = read_npy_file('licks.times.npy')
    lick_ts = TimeSeries(
        name='lick_times',
        timestamps=np.ravel(lick_timestamps),
        data=np.full(len(lick_timestamps), True),
        unit='',
        description='Extracted times of licks, from the lickPiezo signal.'
    )
    lick_bev = BehavioralEvents(lick_ts)
    behavior_module.add_data_interface(lick_bev)


lick_times()
################################################################################
# SPONTANEOUS


def spontaneous():
    """
    Adds spontaneous intervals to acquisition
    Needs data from spontaneous.intervals.npy.
    """
    spont = read_npy_file('spontaneous.intervals.npy')
    spontaneous_ts = TimeIntervals(
        name='spontaneous',
        description='Intervals of sufficient duration when nothing '
                    'else is going on (no task or stimulus presentation'
    )
    for i in range(len(spont[:, 0])):
        spontaneous_ts.add_interval(
            start_time=spont[i, 0],
            stop_time=spont[i, 1],
        )
    nwb_file.add_time_intervals(spontaneous_ts)


spontaneous()
################################################################################
# WHEEL/WHEEL MOVES


def wheel():
    """
    Adds wheel position to nwb.acquisition.
    Needs data from wheel.position.npy and wheel.timestamps.npy.
    """
    wheel_pos = read_npy_file('wheel.position.npy')
    wheel_timestamps = read_npy_file('wheel.timestamps.npy')
    wheel_rate = get_rate(wheel_timestamps)

    wheel_ts = TimeSeries(
        name='wheel_position',
        starting_time=wheel_timestamps[0, 1],
        rate=wheel_rate,
        data=np.ravel(wheel_pos),
        unit='mm',
        conversion=0.135,
        description='The position reading of the rotary encoder attached to '
                    'the rubber wheel that the mouse pushes left and right '
                    'with his forelimbs.',
        comments='The wheel has radius 31 mm and 1440 ticks per revolution, '
                 'so multiply by 2*pi*r/tpr=0.135 to convert to millimeters. '
                 'Positive velocity (increasing numbers) correspond to clockwise '
                 'turns (if looking at the wheel from behind the mouse), i.e. '
                 'turns that are in the correct direction for stimuli presented '
                 'to the left. Likewise negative velocity corresponds to right choices.'
    )
    nwb_file.add_acquisition(wheel_ts)


wheel()


def wheel_moves():
    """
    Adds wheel_moves BehavioralEpochs to behavior processing module.
    Needs data from wheelMoves.type.npy and wheelMoves.intervals.npy.
    """
    wheel_moves_type = read_npy_file('wheelMoves.type.npy')
    wheel_moves_intervals = read_npy_file('wheelMoves.intervals.npy')
    wheel_moves_type = wheel_moves_type.astype(int)  # type cast as int

    wheel_moves_intv = IntervalSeries(
        name='wheel_moves',
        timestamps=np.ravel(wheel_moves_intervals),
        data=np.ravel(wheel_moves_type),
        description='Detected wheel movements.',
        comments='0 for \'flinches\' or otherwise unclassified movements, '
                 '1 for left/clockwise turns, 2 for right/counter-clockwise '
                 'turns (where again "left" means "would be the correct '
                 'direction for a stimulus presented on the left). A detected '
                 'movement is counted as \'left\' or \'right\' only if it was '
                 'sufficient amplitude that it would have registered a correct '
                 'response (and possibly did), within a minimum amount of time '
                 'from the start of the movement. Movements failing those '
                 'criteria are flinch/unclassified type.'
    )
    wheel_moves_be = BehavioralEpochs(wheel_moves_intv)
    behavior_module.add_data_interface(wheel_moves_be)


wheel_moves()
################################################################################
# TRIALS


def trial_table():
    """
    Creates trial table for behavioral trials.
    Needs data from files in trials object.
    """
    # Read data
    included = read_npy_file('trials.included.npy')
    fb_type = read_npy_file('trials.feedbackType.npy')
    fb_type = fb_type.astype(int)
    fb_time = read_npy_file('trials.feedback_times.npy')
    go_cue = read_npy_file('trials.goCue_times.npy')
    trial_intervals = read_npy_file('trials.intervals.npy')
    rep_num = read_npy_file('trials.repNum.npy')
    response_choice = read_npy_file('trials.response_choice.npy')
    response_times = read_npy_file('trials.response_times.npy')
    visual_left = read_npy_file('trials.visualStim_contrastLeft.npy')
    visual_right = read_npy_file('trials.visualStim_contrastRight.npy')
    visual_times = read_npy_file('trials.visualStim_times.npy')

    for j in range(len(trial_intervals)):
        nwb_file.add_trial(trial_intervals[j, 0], trial_intervals[j, 1])

    nwb_file.add_trial_column(
        'included',
        'Importantly, while this variable gives inclusion criteria according '
        'to the definition of disengagement (see manuscript Methods), it does '
        'not give inclusion criteria based on the time of response, as used '
        'for most analyses in the paper.',
        np.ravel(included)
    )
    nwb_file.add_trial_column(
        'go_cue',
        'The \'goCue\' is referred to as the \'auditory tone cue\' in the manuscript.',
        np.ravel(go_cue)
    )
    nwb_file.add_trial_column(
        'visual_stimulus_time',
        'Times are relative to the same time base as every other time in the dataset, '
        'not to the start of the trial.',
        np.ravel(visual_times)
    )
    nwb_file.add_trial_column(
        'visual_stimulus_left_contrast',
        'Proportion contrast. A value of 0.5 means 50% contrast. 0 is a blank '
        'screen: no change to any pixel values on that side (completely undetectable).',
        np.ravel(visual_left)
    )
    nwb_file.add_trial_column(
        'visual_stimulus_right_contrast',
        'Proportion contrast. A value of 0.5 means 50% contrast. 0 is a blank '
        'screen: no change to any pixel values on that side (completely undetectable).',
        np.ravel(visual_right)
    )
    nwb_file.add_trial_column(
        'response_time',
        'Times are relative to the same time base as every other time in the dataset, '
        'not to the start of the trial.',
        np.ravel(response_times)
    )
    nwb_file.add_trial_column(
        'response_choice',
        'Enumerated type. The response registered at the end of the trial, '
        'which determines the feedback according to the contrast condition. '
        'Note that in a small percentage of cases (~4%, see manuscript Methods) '
        'the initial wheel turn was in the opposite direction. -1 for Right '
        'choice (i.e. correct when stimuli are on the right); +1 for left '
        'choice; 0 for Nogo choice.',
        np.ravel(response_choice)
    )
    nwb_file.add_trial_column(
        'feedback_time',
        'Times are relative to the same time base as every other time in the dataset, '
        'not to the start of the trial.',
        np.ravel(fb_time)
    )
    nwb_file.add_trial_column(
        'feedback_type',
        'Enumerated type. -1 for negative feedback (white noise burst); +1 for '
        'positive feedback (water reward delivery).',
        np.ravel(fb_type)
    )
    nwb_file.add_trial_column(
        'rep_num',
        'Trials are repeated if they are "easy" trials (high contrast stimuli '
        'with large difference between the two sides, or the blank screen '
        'condition) and this keeps track of how many times the current '
        'trial\'s condition has been repeated.',
        np.ravel(rep_num)
    )


trial_table()
################################################################################
# STIMULUS


def sparse_noise():
    """
    Adds receptive field mapping task to nwb_file.stimulus.
    Needs data from sparseNoise.positions.npy and sparseNoise.times.npy.
    """
    sparse_noise_pos = read_npy_file('sparseNoise.positions.npy')
    sparse_noise_time = read_npy_file('sparseNoise.times.npy')

    sp_noise = TimeSeries(
        name='receptive_field_mapping_sparse_noise',
        timestamps=np.ravel(sparse_noise_time),
        data=sparse_noise_pos,
        unit='degrees visual angle',
        description='White squares shown on the screen with randomized '
                    'positions and timing - see manuscript Methods.',
        comments='The altitude (first column) and azimuth (second column) '
                 'of the square.'
    )
    nwb_file.add_stimulus(sp_noise)


sparse_noise()


def passive_stimulus():
    """
    Adds passive stimulus replay task to nwb_file.stimulus.
    Needs data from passiveBeeps.times.npy, passiveValveClick.times.npy,
    passiveWhiteNoise.times.npy, and files from the passiveVisual object.
    """
    passive_beeps = read_npy_file('passiveBeeps.times.npy')
    beeps_ts = TimeSeries(
        name='passive_beeps',
        timestamps=np.ravel(passive_beeps),
        data=np.full(len(passive_beeps), True),
        unit='',
        description='Auditory tones of the same frequency as the auditory '
                    'tone cue in the task'
    )
    nwb_file.add_stimulus(beeps_ts)

    passive_clicks = read_npy_file('passiveValveClick.times.npy')
    click_ts = TimeSeries(
        name='passive_click_times',
        timestamps=np.ravel(passive_clicks),
        data=np.full(len(passive_clicks), True),
        unit='',
        description='Opening of the reward valve, but with a clamp in place '
                    'such that no water flows. Therefore the auditory sound of '
                    'the valve is heard, but no water reward is obtained.'
    )
    nwb_file.add_stimulus(click_ts)

    passive_vis_times = read_npy_file('passiveVisual.times.npy')
    passive_vis_left = read_npy_file('passiveVisual.contrastLeft.npy')
    passive_left = TimeSeries(
        name='passive_left_contrast',
        timestamps=np.ravel(passive_vis_times),
        data=np.ravel(passive_vis_left),
        unit='proportion contrast',
        description='Gratings of the same size, spatial freq, position, etc '
                    'as during the discrimination task.'
    )
    nwb_file.add_stimulus(passive_left)

    passive_vis_right = read_npy_file('passiveVisual.contrastRight.npy')
    passive_right = TimeSeries(
        name='passive_right_contrast',
        timestamps=np.ravel(passive_vis_times),
        data=np.ravel(passive_vis_right),
        unit='proportion contrast',
        description='Gratings of the same size, spatial freq, position, etc '
                    'as during the discrimination task.'
    )
    nwb_file.add_stimulus(passive_right)

    passive_noise = read_npy_file('passiveWhiteNoise.times.npy')
    passive_white_noise = TimeSeries(
        name='passive_white_noise',
        timestamps=np.ravel(passive_noise),
        data=np.full(len(passive_noise), True),
        unit='',
        description='The sound that accompanies an incorrect response during the '
                    'discrimination task.'
    )
    nwb_file.add_stimulus(passive_white_noise)


passive_stimulus()
################################################################################
# NEURAL DATA

# DEVICES & ELECTRODE GROUPS
"""
Add in probes as devices and electrode groups.
"""
probe_descriptions = pd.read_csv('probes.description.tsv', sep='\t')
probe_descriptions = list(probe_descriptions['description'])
electrode_groups = list()
for i in range(len(probe_descriptions)):
    probe_device = Device(name=str(i))
    probe_electrode_group = ElectrodeGroup(
        name='Probe' + str(i + 1),
        description='Neuropixels Phase3A opt3',
        device=probe_device,
        location=''
    )
    nwb_file.add_device(probe_device)
    electrode_groups.append(probe_electrode_group)
    nwb_file.add_electrode_group(probe_electrode_group)

# CHANNELS
"""
Add channel information into the Electrode Table.
"""
# Read data
insertion_df = pd.read_csv('probes.insertion.tsv', sep='\t')
insertion_df['group_name'] = insertion_df.index.values

channel_site = read_npy_file('channels.site.npy')
channel_brain = pd.read_csv('channels.brainLocation.tsv', sep='\t')

channel_probes = read_npy_file('channels.probe.npy')
channel_probes = np.ravel(channel_probes.astype(int))
channel_table = pd.DataFrame(columns=['group_name'])
channel_table['group_name'] = channel_probes
channel_table = channel_table.merge(insertion_df, 'left', 'group_name')

entry_point_rl = np.array(channel_table['entry_point_rl'])
entry_point_ap = np.array(channel_table['entry_point_ap'])
axial_angle = np.array(channel_table['axial_angle'])
vertical_angle = np.array(channel_table['vertical_angle'])
horizontal_angle = np.array(channel_table['horizontal_angle'])
distance_advanced = np.array(channel_table['distance_advanced'])

locations = np.array(channel_brain['allen_ontology'])
groups = np.asarray([electrode_groups[c] for c in channel_probes])
channel_site_pos = read_npy_file('channels.sitePositions.npy')

for i in range(len(groups)):
    nwb_file.add_electrode(
        x=float('NaN'),
        y=float('NaN'),
        z=float('NaN'),
        imp=float('NaN'),
        location=str(locations[i]),
        group=groups[i],
        filtering='none'
    )

# Add Electrode columns
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
    name='ccf_ap',
    description='The AP position in Allen Institute\'s Common Coordinate Framework.',
    data=np.array(channel_brain['ccf_ap'])
)
nwb_file.add_electrode_column(
    name='ccf_dv',
    description='The DV position in Allen Institute\'s Common Coordinate Framework.',
    data=np.array(channel_brain['ccf_dv'])
)
nwb_file.add_electrode_column(
    name='ccf_lr',
    description='The LR position in Allen Institute\'s Common Coordinate Framework.',
    data=np.array(channel_brain['ccf_lr'])
)

# Insertion
nwb_file.add_electrode_column(
    name='entry_point_rl',
    description='mediolateral position of probe entry point relative to midline (microns). '
                'Positive means right',
    data=entry_point_rl
)
nwb_file.add_electrode_column(
    name='entry_point_ap',
    description='anteroposterior position of probe entry point relative to bregma (microns). '
                'Positive means anterior',
    data=entry_point_ap
)
nwb_file.add_electrode_column(
    name='vertical_angle',
    description='vertical angle of probe (degrees). Zero means horizontal. '
                'Positive means pointing down',
    data=vertical_angle
)
nwb_file.add_electrode_column(
    name='horizontal_angle',
    description='horizontal angle of probe (degrees), after vertical rotation. '
                'Zero means anterior. Positive means counterclockwise (i.e. left).',
    data=horizontal_angle
)
nwb_file.add_electrode_column(
    name='axial_angle',
    description='axial angle of probe (degrees). Zero means that without vertical and horizontal rotations, '
                'the probe contacts would be pointing up. Positive means "counterclockwise.',
    data=axial_angle
)
nwb_file.add_electrode_column(
    name='distance_advanced',
    description='How far the probe was moved forward from its entry point. (microns).',
    data=distance_advanced
)
# CLUSTERS & SPIKES
"""
Add cluster information into the Unit Table.
"""
# Read data
cluster_probe = read_npy_file('clusters.probes.npy')
cluster_probe = np.ravel(cluster_probe.astype(int))
cluster_channel = read_npy_file('clusters.peakChannel.npy')
cluster_depths = read_npy_file('clusters.depths.npy')
phy_annotations = np.ravel(read_npy_file('clusters._phy_annotation.npy'))
waveform_chans = read_npy_file('clusters.templateWaveformChans.npy')
waveform_chans = waveform_chans.astype(int)
waveform = read_npy_file('clusters.templateWaveforms.npy')
waveform_duration = read_npy_file('clusters.waveformDuration.npy')
spike_to_clusters = read_npy_file('spikes.clusters.npy')
spike_times = read_npy_file('spikes.times.npy')
spike_amps = read_npy_file('spikes.amps.npy')
spike_depths = read_npy_file('spikes.depths.npy')

# Sorting spikes into clusters
cluster_info = dict()
for i in range(len(spike_to_clusters)):
    s = int(spike_to_clusters[i])
    if s not in cluster_info:
        cluster_info[s] = [i]
    else:
        cluster_info[s].append(i)

# Add Unit Columns
nwb_file.add_unit_column(
    name='peak_channel',
    description='The channel number of the location of the peak of the cluster\'s waveform.'
)
nwb_file.add_unit_column(
    name='waveform_duration',
    description='The trough-to-peak duration of the waveform on the peak channel.'
)
nwb_file.add_unit_column(
    name='phy_annotations',
    description='0 = noise (these are already excluded and don\'t appear in this dataset '
                'at all); 1 = MUA (i.e. presumed to contain spikes from multiple neurons; '
                'these are not analyzed in any analyses in the paper); 2 = Good (manually '
                'labeled); 3 = Unsorted. In this dataset \'Good\' was applied in a few but '
                'not all datasets to included neurons, so in general the neurons with '
                '_phy_annotation>=2 are the ones that should be included.',
)
nwb_file.add_unit_column(
    name='cluster_depths',
    description='The position of the center of mass of the template of the cluster, '
                'relative to the probe. The deepest channel on the probe is depth=0, '
                'and the most superficial is depth=3820. Units: µm',
)
nwb_file.add_unit_column(
    name='sampling_rate',
    description='Sampling rate, in Hz.',
)

# Add Units by cluster
for i in cluster_info:
    c = cluster_info[i]
    times = np.array(spike_times[c])
    annotations = phy_annotations[i]
    annotations = annotations.astype(int)
    channel = cluster_channel[i]
    channel = channel.astype(int)
    duration = waveform_duration[i]
    duration = duration.astype(int)

    nwb_file.add_unit(
        spike_times=np.ravel(times),
        electrodes=waveform_chans[i, :],
        electrode_group=electrode_groups[cluster_probe[i]],
        waveform_mean=waveform[i, :, :],
        id=i,
        phy_annotations=annotations,
        peak_channel=channel,
        waveform_duration=duration,
        cluster_depths=cluster_depths[i],
        sampling_rate=30000.0
    )

# Add spike amps and depths
amps = {}
depths = {}

for c in cluster_info.keys():
    amps[c] = spike_amps[cluster_info[c]]
    depths[c] = spike_depths[cluster_info[c]]

add_ragged_data_to_dynamic_table(
    table=nwb_file.units,
    data=amps,
    column_name='spike_amps',
    column_description='The peak-to-trough amplitude, obtained from the template and '
                'template-scaling amplitude returned by Kilosort (not from the raw data).'
)
add_ragged_data_to_dynamic_table(
    table=nwb_file.units,
    data=depths,
    column_name='spike_depths',
    column_description='The position of the center of mass of the spike on the probe, '
                       'determined from the principal component features returned by Kilosort. '
                       'The deepest channel on the probe is depth=0, and the most superficial is depth=3820.'
)
################################################################################
# WRITE TO FILE

with NWBHDF5IO('test_build_nwb_file.nwb', 'w') as io:
    io.write(nwb_file)
    print('saved')
