from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile, NWBHDF5IO
import pynwb
import numpy as np

################################################################################
# CREATE FILE
nwb_file = NWBFile(
    session_description='Test to see if building a file works',
    identifier='Test1',
    session_start_time=datetime(2016, 12, 14, tzinfo=tzlocal()),
    file_create_date=datetime.now(tzlocal())
)
behavioral_ts = pynwb.behavior.BehavioralTimeSeries()
behavioral_events = pynwb.behavior.BehavioralEvents()
position = pynwb.behavior.Position()
pupil_track = pynwb.behavior.PupilTracking()
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
    Eye PupilTracking: Eye-tracking data for pupil size
    :param timestamps: numpy array
    :param area: numpy array, area of pupil
    :param xy_pos: numpy array, XY positions for pupil
    """
    nwb_file.add_acquisition(pupil_track.create_timeseries(
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
    ))
    nwb_file.add_acquisition(position.create_spatial_series(
        name='xy_positions',
        timestamps=timestamps,
        data=xy_pos,  # currently as [x, y] pairs
        reference_frame='Video frame',
        description='Features extracted from the video of the right eye.',
        comments='The 2D position of the center of the pupil in the video '
                 'frame. This is not registered to degrees visual angle, but '
                 'could be used to detect saccades or other changes in eye position.'
    ))


eye_nwb(eye_timestamps, eye_area, eye_xyPos)
################################################################################
# FACE

face_motionEnergy = read_npy_file('face.motionEnergy.npy')
face_timestamps = read_npy_file('face.timestamps.npy')


def face_nwb(motion_energy, timestamps):
    """
    Behavioral Time Series for face
    Timestamps are interpolated between two points in second column of array
    """
    timestamps = interpol_timestamps(timestamps)
    nwb_file.add_acquisition(behavioral_ts.create_timeseries(
        name='face_motion_energy',
        timestamps=timestamps,
        data=np.ravel(motion_energy),
        unit='arb. unit',
        description='Features extracted from the video of the frontal aspect of '
                    'the subject, including the subject\'s face and forearms.',
        comments='The integrated motion energy across the whole frame, i.e. '
                 'sum( (thisFrame-lastFrame)^2 ). Some smoothing is applied '
                 'before this operation.'
    ))


face_nwb(face_motionEnergy, face_timestamps)
################################################################################
# LICK_PIEZO/LICKS

lickP_raw = read_npy_file('lickPiezo.raw.npy')
lickP_timestamps = read_npy_file('lickPiezo.timestamps.npy')


def lick_piezo(arr, timestamps):
    """
    Behavioral Time Series for Piezo licks
    Timestamps get interpolated
    """
    timestamps = interpol_timestamps(timestamps)
    nwb_file.add_acquisition(behavioral_ts.create_timeseries(
        name='lickPiezo',
        timestamps=timestamps,
        data=np.ravel(arr),
        unit='V',
        description='Voltage values from a thin-film piezo connected to the '
                    'lick spout, so that values are proportional to deflection '
                    'of the spout and licks can be detected as peaks of the signal.'
    ))


lick_piezo(lickP_raw, lickP_timestamps)
lick_times_np = read_npy_file('licks.times.npy')


def lick_times(times):
    """
    :param times: numpy array of lick timing
    """
    nwb_file.add_acquisition(behavioral_events.create_timeseries(
        name='lick_times',
        timestamps=np.ravel(times),
        description='Extracted times of licks, from the lickPiezo signal.'
    ))


lick_times(lick_times_np)
################################################################################
# WHEEL/WHEEL MOVES

wheel_pos = read_npy_file('wheel.position.npy')
wheel_timestamps = read_npy_file('wheel.timestamps.npy')


def wheel(pos, timestamps):
    """
    wheel position
    :param pos: numpy array
    :param timestamps: numpy array
    """
    timestamps = interpol_timestamps(timestamps)
    nwb_file.add_acquisition(behavioral_ts.create_timeseries(
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
    ))


wheel(wheel_pos, wheel_timestamps)
################################################################################
# WRITE TO FILE

nwb_file.add_acquisition(behavioral_ts)
nwb_file.add_acquisition(behavioral_events)
nwb_file.add_acquisition(position)
nwb_file.add_acquisition(pupil_track)

with NWBHDF5IO('test_build_nwb_file.nwb', 'w') as io:
    io.write(nwb_file)
    print('saved')
