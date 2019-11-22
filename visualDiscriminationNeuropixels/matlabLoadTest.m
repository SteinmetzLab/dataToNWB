

% Script to test Renee's NWB file

%% 

filename = 'D:\DownloadsD\test_build_nwb_file (1).nwb';

%%

stimOn = h5read(filename, '/intervals/trials/visual_stimulus_time');
conL = h5read(filename, '/intervals/trials/visual_stimulus_left_contrast');
conR = h5read(filename, '/intervals/trials/visual_stimulus_right_contrast');
choice = h5read(filename, '/intervals/trials/response_choice');
fb = h5read(filename, '/intervals/trials/feedback_type');

[stimOn(1:10) conL(1:10) conR(1:10) choice(1:10) fb(1:10)]

whos stimOn conL conR choice fb

%%

