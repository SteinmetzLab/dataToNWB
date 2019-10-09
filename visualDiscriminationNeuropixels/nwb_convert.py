from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import NWBFile
import tarfile
import pynwb
import numpy as np
import pandas as pd


################################################################################
# CREATE FILE
nwb_file = NWBFile(
    session_description='test',
    identifier='',
    session_start_time=datetime(2016, 12, 14),
    file_create_date=datetime.now(tzlocal())
)
################################################################################


def read_npy_file(filename):
    """
    :param filename: loads filename
    :return: file as numpy array
    currently prints first 3 rows  # TODO: remove printing when done
    """
    np_arr = np.load(filename)
    print(np_arr.shape)
    print(filename, np_arr[:3, :])
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
# eye_nwb(eye_timestamps, eye_area, eye_xyPos)


def eye_nwb(timestamps, area, xy_pos):
    """
    Eye
    PupilTracking: Eye-tracking data for pupil size
    """
    eye_area_pupiltrack = pynwb.behavior.PupilTracking()
    eye_area_pupiltrack.create_timeseries(
        name='eye_area',
        timestamps=timestamps,
        data=np.ravel(area),
        units='AU',
        description='Insert description'
    )
    nwb_file.add_acquisition(eye_area_pupiltrack)

    eye_position = pynwb.behavior.Position()
    eye_position.create_spatial_series(
        name='xy_positions',
        timestamps=timestamps,
        data=xy_pos,  # currently as [x, y] pairs
        units='AU',
        description='Insert description'
    )
    nwb_file.add_acquisition(eye_position)


################################################################################
# FACE

face_motionEnergy = read_npy_file('face.motionEnergy.npy')
face_timestamps = read_npy_file('face.timestamps.npy')


def face_nwb(motion_energy, timestamps):
    """
    Behavioral Time Series for face
    Timestamps are interpolated between two points in second column of array
    """
    interpol_timestamps(timestamps)
    behavior_face = pynwb.behavior.BehavioralTimeSeries()
    behavior_face.create_timeseries(
        name='face_motion_energy',
        timestamps=timestamps,
        data=np.ravel(motion_energy),
        units='J?',  # TODO: find out units
        description='Put in description here'
    )
    nwb_file.add_acquisition(behavior_face)


################################################################################
# LICKPIEZO

lickP_raw = read_npy_file('lickPiezo.raw.npy')
lickP_timestamps = read_npy_file('lickPiezo.timestamps.npy')


def lick_piezo(arr, timestamps):
    """
    Behavioral Time Series for Piezo licks
    Timestamps get interpolated
    """
    timestamps = interpol_timestamps(timestamps)
    lick_p = pynwb.behavior.BehavioralTimeSeries()
    lick_p.create_timeseries(
        name='lick_Piezo',
        timestamps=timestamps,
        data=np.ravel(arr),
        units='V',   # voltage
        description='Insert description'
    )
    nwb_file.add_acquisition(lick_p)


lick_times = read_npy_file('licks.times.npy')


def lick_times(times):
    """
    :param times: numpy array of lick timing
    """
    lick_times = pynwb.behavior.BehavioralEvents()
    lick_times.create_timeseries(
        name='lick_times',
        timestamps=np.ravel(times),
        description='timing of licks'
    )
    nwb_file.add_acquisition(lick_times)


################################################################################
# WHEEL

wheel_pos = read_npy_file('wheel.position.npy')
wheel_timestamps = read_npy_file('wheel.timestamps.npy')


def wheel(pos, timestamps):
    """
    wheel position
    :param pos: numpy array
    :param timestamps: numpy array
    """
    timestamps = interpol_timestamps(timestamps)
    wheel_bts = pynwb.behavior.BehavioralTimeSeries()
    wheel_bts.add_timeseries(
        name='wheel_position',
        timestamps=timestamps,
        data=np.ravel(pos),
        units='mm?',  # TODO: find out units
        description='insert description'
    )
    nwb_file.add_acquisition(wheel_bts)


wheelMoves_intervals = read_npy_file('wheelMoves.intervals.npy')
wheelMoves_type = read_npy_file('wheelMoves.type.npy')


def wheel_moves(types, intervals):
    """
    wheel moves
    :param types: numpy array
    :param intervals: numpy array
    """
    wheel_m = pynwb.behavior.BehavioralEpochs()
    wheel_m.create_interval_series(
        name='wheel_moves',
        timestamps=np.ravel(intervals),  # start/stop separated by space
        data=np.ravel(types),
        description='Insert description'
    )
    nwb_file.add_acquisition(wheel_m)


################################################################################
# WRITE TO FILE
"""
def write_file(filename, data):
    start_time = datetime(2019, 9, 27, tzinfo=tzlocal())
    create_date = datetime(2019, 9, 27, tzinfo=tzlocal())
    nwbfile = NWBFile(session_description='testing',
                      identifier='NWB123',
                      session_start_time=start_time,
                      file_create_date = create_date)

    test_ts = TimeSeries(name='synthetic_timeseries',
                         data=data,
                         unit='SIunit',
                         rate=1.0,
                         starting_time=0.0)
    nwbfile.add_acquisition(test_ts)

    io = NWBHDF5IO(filename, 'w')
    io.write(nwbfile)
    io.close()
"""
