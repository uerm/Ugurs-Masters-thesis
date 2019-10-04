%% Load data
%clear,clc
% Long Term AF Database
% The first 7 recordings are discarded since they cannot be loaded.
% The record names/numbers have been changed for some records to remove
% the leading zeros.
Data2 = [10, 11, 12, 13, 15, 16, 17, 18, 19, 20,...
    21, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34, 35, 37, 38, 39, 42, 43, 44,...
    45, 47, 48, 49, 51, 53, 54, 55, 56, 58, 60, 62, 64, 65, 68, 69, 70, 71,...
    72, 74, 75, 100, 101, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115,...
    116, 117, 118, 119, 120, 121, 122, 200, 201, 202, 203, 204, 205, 206,...
    207, 208];

for i = 1:length(Data2)
    [s1{i}, Fs1, t1{i}] = rdsamp(num2str(Data2(i)),1); % Lead I
    [s2{i}, Fs2, t2{i}] = rdsamp(num2str(Data2(i)),2); % Lead II
end
%% Find peaks and boundaries

for i = 1:length(Data2)
    ecgpuwave(num2str(Data2(i)),'test');
    pwaves1{i} = rdann(num2str(Data2(i)),'test',[],[],[],'p');
    twaves1{i} = rdann(num2str(Data2(i)),'test',[],[],[],'t');
    QRS1{i} = rdann(num2str(Data2(i)),'test',[],[],[],'N');
    [wave1{i},loc1{i}] = rdann(num2str(Data2(i)),'test',[],[],[],'');
end

%% Plot raw signal
subplot(211)
plot(t1{1,40},s1{1,40})
xlim([0 40])
subplot(212)
plot(t2{1,40},s2{1,40})
xlim([0 40])
