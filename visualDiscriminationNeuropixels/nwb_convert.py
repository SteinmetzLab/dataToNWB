from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.behavior import BehavioralEvents, BehavioralEpochs, BehavioralTimeSeries, Position, PupilTracking, \
    IntervalSeries
from pynwb.epoch import TimeIntervals
import numpy as np

################################################################################
# CREATE FILE
nwb_file = NWBFile(
    session_description='Test to see if building a file works',
    identifier='Test123',
    session_start_time=datetime(2016, 12, 14, tzinfo=tzlocal()),
    file_create_date=datetime.now(tzlocal()),
    experimenter='name',
    experiment_description='description',
    institution='institution'  # add/fill out for real file
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


def interpol_timestamps(timestamps):
    """
    Gets full timestamps array for (2, 2) timestamps array
    :param timestamps: (2, 2) numpy array
    :return: timestamps as numpy array
    """
    return np.linspace(timestamps[0, 1], timestamps[1, 1], timestamps[1, 0] + 1)


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
    face_timestamps = interpol_timestamps(face_timestamps)
    face_energy = TimeSeries(
        name='face_motion_energy',
        timestamps=face_timestamps,
        data=np.ravel(face_motion_energy),
        unit='arb. unit',
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
    lp_timestamps = interpol_timestamps(lp_timestamps)
    lick_piezo_ts = TimeSeries(
        name='lickPiezo',
        timestamps=lp_timestamps,
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
    wheel_timestamps = interpol_timestamps(wheel_timestamps)

    wheel_ts = TimeSeries(
        name='wheel_position',
        timestamps=wheel_timestamps,
        data=np.ravel(wheel_pos),
        unit='encoder ticks',
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
    fb_time = read_npy_file('trials.feedback_times.npy')
    go_cue = read_npy_file('trials.goCue_times.npy')
    trial_intervals = read_npy_file('trials.intervals.npy')
    rep_num = read_npy_file('trials.repNum.npy')
    response_choice = read_npy_file('trials.response_choice.npy')
    response_times = read_npy_file('trials.response_times.npy')
    visual_left = read_npy_file('trials.visualStim_contrastLeft.npy')
    visual_right = read_npy_file('trials.visualStim_contrastRight.npy')
    visual_times = read_npy_file('trials.visualStim_times.npy')

    for i in range(len(trial_intervals)):
        nwb_file.add_trial(trial_intervals[i, 0], trial_intervals[i, 1])
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
# print(nwb_file.trials.to_dataframe())
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
        description='Auditory tones of the same frequency as the auditory '
                    'tone cue in the task'
    )
    nwb_file.add_stimulus(beeps_ts)

    passive_clicks = read_npy_file('passiveValveClick.times.npy')
    click_ts = TimeSeries(
        name='passive_click_times',
        timestamps=np.ravel(passive_clicks),
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
        description='The sound that accompanies an incorrect response during the '
                    'discrimination task.'
    )
    nwb_file.add_stimulus(passive_white_noise)


passive_stimulus()
################################################################################
# WRITE TO FILE


with NWBHDF5IO('test_build_nwb_file.nwb', 'w') as io:
    io.write(nwb_file)
    print('saved')
