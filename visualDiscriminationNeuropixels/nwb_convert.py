from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO, TimeSeries, ProcessingModule
from pynwb.behavior import BehavioralEvents, BehavioralEpochs, BehavioralTimeSeries, \
    Position, PupilTracking, IntervalSeries, SpatialSeries
import numpy as np

################################################################################
# CREATE FILE

nwb_file = NWBFile(
    session_description='Test to see if building a file works',
    identifier='Test1',
    session_start_time=datetime(2016, 12, 14, tzinfo=tzlocal()),
    file_create_date=datetime.now(tzlocal())
)
################################################################################
# PROCESS DATA


def read_npy_file(filename):
    """
    :param filename: filename being loaded, .npy file
    :return: file as numpy array
    """
    np_arr = np.load(filename)
    return np_arr


def interpol_timestamps(timestamps):
    """
    :param timestamps: (2, 2) numpy array
    :return: Interpolated timestamps as numpy array
    """
    return np.linspace(timestamps[0, 1], timestamps[1, 1], timestamps[1, 0] + 1)


################################################################################
# EYE

eye_timestamps = read_npy_file('eye.timestamps.npy')[:, 1]
eye_area = read_npy_file('eye.area.npy')
eye_xyPos = read_npy_file('eye.xyPos.npy')


def eye_nwb(timestamps, area, xy_pos):
    """
    Eye ProcessingModule with Pupil TimeSeries and Position SpatialSeries
    :param timestamps: numpy array
    :param area: numpy array, area of pupil
    :param xy_pos: numpy array, XY positions for pupil
    """
    eye_module = ProcessingModule('eye', 'Features extracted from the video of the right eye.')
    nwb_file.add_processing_module(eye_module)
    pupil = TimeSeries(
        name='eye_area',
        timestamps=timestamps,
        data=np.ravel(area),
        unit='arb. unit',
        description='Features extracted from the video of the right eye.',
        comments='The area of the pupil extracted with DeepLabCut. Note that '
                 'it is relatively very small during the discrimination task '
                 'and during the passive replay because the three screens are '
                 'medium-grey at this time and black elsewhere - so the much '
                 'brighter overall luminance levels lead to relatively '
                 'constricted pupils.'
    )
    eye_xy = SpatialSeries(
        name='eye_xy_positions',
        timestamps=timestamps,
        data=xy_pos,  # currently as [x, y] pairs
        reference_frame='Video frame',
        description='Features extracted from the video of the right eye.',
        comments='The 2D position of the center of the pupil in the video '
                 'frame. This is not registered to degrees visual angle, but '
                 'could be used to detect saccades or other changes in eye position.'
    )
    position = Position(eye_xy)
    pupil_track = PupilTracking(pupil)
    eye_module.add_data_interface(position)
    eye_module.add_data_interface(pupil_track)


eye_nwb(eye_timestamps, eye_area, eye_xyPos)
################################################################################
# FACE

face_motionEnergy = read_npy_file('face.motionEnergy.npy')
face_timestamps = read_npy_file('face.timestamps.npy')


def face_nwb(motion_energy, timestamps):
    """
    Face ProcessingModule with Face Energy BehavioralTimeSeries
    :param motion_energy: numpy array
    :param timestamps: numpy array, timestamps are interpolated between two points in second column of array
    """
    face_module = ProcessingModule('face', 'Features extracted from the video of the frontal aspect '
                                           'of the subject, including the subject\'s face and forearms.')
    nwb_file.add_processing_module(face_module)
    timestamps = interpol_timestamps(timestamps)
    face_energy = TimeSeries(
        name='face_motion_energy',
        timestamps=timestamps,
        data=np.ravel(motion_energy),
        unit='arb. unit',
        description='Features extracted from the video of the frontal aspect of '
                    'the subject, including the subject\'s face and forearms.',
        comments='The integrated motion energy across the whole frame, i.e. '
                 'sum( (thisFrame-lastFrame)^2 ). Some smoothing is applied '
                 'before this operation.'
    )
    face_interface = BehavioralTimeSeries(face_energy)
    face_module.add_data_interface(face_interface)


face_nwb(face_motionEnergy, face_timestamps)
################################################################################
# LICK_PIEZO/LICKS

lickP_raw = read_npy_file('lickPiezo.raw.npy')
lickP_timestamps = read_npy_file('lickPiezo.timestamps.npy')


def lick_piezo(arr, timestamps):
    """
    Acquisition TimeSeries for Piezo licks
    :param arr: numpy array (data)
    :param timestamps: timestamps are interpolated between two points in second column of numpy array
    """
    timestamps = interpol_timestamps(timestamps)
    lick_piezo_ts = TimeSeries(
        name='lickPiezo',
        timestamps=timestamps,
        data=np.ravel(arr),
        unit='V',
        description='Voltage values from a thin-film piezo connected to the '
                    'lick spout, so that values are proportional to deflection '
                    'of the spout and licks can be detected as peaks of the signal.'
    )
    nwb_file.add_acquisition(lick_piezo_ts)


lick_piezo(lickP_raw, lickP_timestamps)
lick_times_np = read_npy_file('licks.times.npy')


def lick_times(times):
    """
    Lick ProcessingModule for Lick BehavioralEvents
    :param times: :param times: numpy array of lick timing
    """
    lick_module = ProcessingModule('lick', 'Extracted times of licks, from the lickPiezo signal.')
    nwb_file.add_processing_module(lick_module)
    lick_ts = TimeSeries(
        name='lick_times',
        timestamps=np.ravel(times),
        description='Extracted times of licks, from the lickPiezo signal.'
    )
    lick_bev = BehavioralEvents(lick_ts)
    lick_module.add_data_interface(lick_bev)


lick_times(lick_times_np)


################################################################################
# SPONTANEOUS


def spontaneous(timestamps):
    """
    Acquisition for Spontaneous TimeSeries
    :param timestamps: numpy array
    """
    spontaneous_ts = TimeSeries(
        name='spontaneous',
        timestamps=np.ravel(timestamps),
        description='Intervals of sufficient duration when nothing '
                    'else is going on (no task or stimulus presentation'
    )
    nwb_file.add_acquisition(spontaneous_ts)


spont = read_npy_file('spontaneous.intervals.npy')
spontaneous(spont)
################################################################################
# WHEEL/WHEEL MOVES

wheel_pos = read_npy_file('wheel.position.npy')
wheel_timestamps = read_npy_file('wheel.timestamps.npy')


def wheel(pos, timestamps):
    """
    Acquisition for wheel position TimeSeries
    :param pos: numpy array
    :param timestamps: timestamps are interpolated between two points in second column of numpy array
    """
    timestamps = interpol_timestamps(timestamps)
    wheel_ts = TimeSeries(
        name='wheel_position',
        timestamps=timestamps,
        data=np.ravel(pos),
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


wheel(wheel_pos, wheel_timestamps)

wheelMoves_type = read_npy_file('wheelMoves.type.npy')
wheelMoves_intervals = read_npy_file('wheelMoves.intervals.npy')
wheelMoves_intervals = wheelMoves_intervals.astype(int)


def wheel_moves(types, intervals):
    """
    Wheel Moves ProcessingModule with Wheel Moves BehavioralEpochs
    :param types: numpy array
    :param intervals: numpy array, start/stop times separated by space
    """
    types = types.astype(int)
    wheel_m_module = ProcessingModule('wheelMoves', '')
    nwb_file.add_processing_module(wheel_m_module)
    wheel_moves_intv = IntervalSeries(
        name='wheel_moves',
        timestamps=np.ravel(intervals),
        data=np.ravel(types),
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
    wheel_m_module.add_data_interface(wheel_moves_be)


wheel_moves(wheelMoves_type, wheelMoves_intervals)


################################################################################
# WRITE TO FILE

with NWBHDF5IO('test_build_nwb_file.nwb', 'w') as io:
    io.write(nwb_file)
    print('saved')
