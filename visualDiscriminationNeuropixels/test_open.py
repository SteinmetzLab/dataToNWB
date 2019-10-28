"""
Test file for opening a NWB file.
"""
from pynwb import NWBHDF5IO

io = NWBHDF5IO('test_build_nwb_file.nwb', 'r')
nwb_file_in = io.read()
print(nwb_file_in)

# acquisition
wheel_pos = nwb_file_in.acquisition['wheel_position']
print(wheel_pos)
print(wheel_pos.data[:])  # get actual data array

# processing
behavior = nwb_file_in.processing['behavior']
print(behavior)
print(behavior['Position'])

# trials table
intervals = nwb_file_in.intervals
print(intervals['trials'].to_dataframe())
print(intervals['trials']['feedback_type'].description)
