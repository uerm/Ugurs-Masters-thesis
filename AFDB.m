%% Load ECG signals
clear,clc



% AF database
% The record names/numbers have been changed for all records to remove
% the leading zeros.

Data1 = [04015, 04043, 04048, 04126, 04746, 04908, 04936, 05091, 05121, 05261 ...
    06426, 06453, 06995, 07162, 07859, 07879, 07910, 08215, 08219, 08378,...
    08405, 08434, 08455];


for i = 1:length(Data1)
    [sig_1{i}, Fs1, tm_1{i}] = rdsamp(num2str(Data1(i)),1); % Lead I
    [sig_2{i}, Fs2, tm_2{i}] = rdsamp(num2str(Data1(i)),2); % Lead II
end