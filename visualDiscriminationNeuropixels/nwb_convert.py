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
################################################################################


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
    wheel_bts = pynwb.behavior.BehavioralTimeSeries()
    wheel_bts.create_timeseries(
        name='wheel_position',
        timestamps=timestamps,
        data=np.ravel(pos),
        unit='encoder ticks',
        description='The position reading of the rotary encoder attached to '
                    'the rubber wheel that the mouse pushes left and right '
                    'with his forelimbs.',
        comments="""The wheel has radius 31 mm and 1440 ticks per revolution, \
        so multiply by 2*pi*r/tpr=0.135 to convert to millimeters. Positive \
        velocity (increasing numbers) correspond to clockwise turns \
        (if looking at the wheel from behind the mouse), i.e. turns that \
        are in the correct direction for stimuli presented to the left. \
        Likewise negative velocity corresponds to right choices."""
    )
    nwb_file.add_acquisition(wheel_bts)


wheel(wheel_pos, wheel_timestamps)
################################################################################
# WRITE TO FILE

with NWBHDF5IO('test_build_nwb_file.nwb', 'w') as io:
    io.write(nwb_file)
    print('saved')