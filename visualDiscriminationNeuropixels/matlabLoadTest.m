

% Script to test Renee's NWB file

%% 
clear all
filename = 'Z:\Subjects\UCL_data\NWB\Files11.26\Steinmetz2019_Hench_2017-06-16.nwb';

origFiles = 'D:\spikeAndBehavioralData\Hench_2017-06-16';

%%
s = loadSession(origFiles)

%%

stimOn = h5read(filename, '/intervals/trials/visual_stimulus_time');
conL = h5read(filename, '/intervals/trials/visual_stimulus_left_contrast');
conR = h5read(filename, '/intervals/trials/visual_stimulus_right_contrast');
choice = h5read(filename, '/intervals/trials/response_choice');
fb = h5read(filename, '/intervals/trials/feedback_type');

[stimOn(1:10) conL(1:10) conR(1:10) choice(1:10) fb(1:10)]

whos stimOn conL conR choice fb

trCheck = all(stimOn==s.trials.visualStim_times) & ...
    all(conL==s.trials.visualStim_contrastLeft) & ...
    all(choice==s.trials.response_choice) & ...
    all(fb == s.trials.feedbackType);

fprintf(1, 'behavioral data: %s\n', mat2str(trCheck))


%% load spike times


st = h5read(filename, '/units/spike_times');
stidx = h5read(filename, '/units/spike_times_index');
id = h5read(filename, '/units/id');


st1_nwb = st(stidx(find(id==1)-1)+1:stidx(find(id==1)));

stCheck = all(st1_nwb==s.spikes.times(s.spikes.clusters==1));

fprintf(1, 'spike times in cluster 1: %s\n', mat2str(stCheck))


%% cluster properties

cd = h5read(filename, '/units/cluster_depths')';

[sID, ii] = sort(id); 

cdCheck = all(cd(ii)==s.clusters.depths);

fprintf(1, 'cluster depths: %s\n', mat2str(cdCheck))

%% passive

pb = h5read(filename, '/stimulus/presentation/passive_beeps/timestamps');

passCheck = all(pb==s.passiveBeeps.times);

fprintf(1, 'passive stimulus: %s\n', mat2str(passCheck))
