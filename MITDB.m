%% Load ECG signals
clear,clc

% MITDB Data
Data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,...
    114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203,...
    205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222,...
    223, 228, 230, 231, 232, 233, 234];

% AF database
% Data = [04015, 04043, 04048, 04126, 04746, 04908, 04936, 05091, 05121, 05261 ...
%     06426, 06453, 06995, 07162, 07859, 07879, 07910, 08215, 08219, 08378,...
%     08405, 08434, 08455];

% Long Term AF Database
% Data = [00, 01, 03, 05, 06, 07, 08, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20,...
%     21, 22, 23, 24, 25, 26, 28, 30, 32, 33, 34, 35, 37, 38, 39, 42, 43, 44,...
%     45, 47, 48, 49, 51, 53, 54, 55, 56, 58, 60, 62, 64, 65, 68, 69, 70, 71,...
%     72, 74, 75, 100, 101, 102, 103, 104, 105, 110, 111, 112, 113, 114, 115,...
%     116, 117, 118, 119, 120, 121, 122, 200, 201, 202, 203, 204, 205, 206,...
%     207, 208];


% Lead I
for i = 1:length(Data)
    [sig1{i}, Fs1, tm1{i}] = rdsamp(strcat('mitdb/', num2str(Data(i))),1);
end

% Lead II
for j = 1:length(Data)
    [sig2{j}, Fs2, tm2{j}] = rdsamp(strcat('mitdb/', num2str(Data(j))),2);
end

%% Read annotations

for i = 1:length(Data)
    [ann{i}, anntype{i}] = rdann(strcat('mitdb/',num2str(Data(i))),'atr');
end

%% Find waves and QRS complex

for i = 1:2
    for j = 1:length(Data)
        ecgpuwave(num2str(Data(j)),'test');
        [signal{i,j},Fs,time{j}] = rdsamp(num2str(Data(j)));
        pwaves{i,j} = rdann(num2str(Data(j)),'test',[],[],[],'p');
        twaves{i,j} = rdann(num2str(Data(j)),'test',[],[],[],'t');
        QRS{i,j} = rdann(num2str(Data(j)),'test',[],[],[],'N');
        [wave{i,j},loc{i,j}] = rdann(num2str(Data(j)),'test',[],[],[],'');
    end
end

%% Plot of subject 48 - both leads
subplot(211)
plot(tm1{1,1},sig1{1,48})
title('Subject 48, lead I')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')

subplot(212)
plot(tm2{1,1},sig2{1,48})
title('Subject 48, lead II')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')

%% Denoising with DWT - Lead I
y1 = dwt_denoise1(sig1); % Denoising Lead I
y2 = dwt_denoise2(sig2); % Denoising Lead II


%% Result of DWT filtering

subplot(221)
plot(tm1{1,48},sig1{1,48})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead I')
legend('No filtering','Location','Best')


subplot(222)
plot(tm2{1,48},sig2{1,48})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead II')
legend('No filtering','Location','Best')


subplot(223)
plot(tm1{1,48}, y1{1,48}*(-1))
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead I')
legend('DWT filtering','Location','Best')

subplot(224)
plot(tm2{1,48}, y2{1,48}*(-1))
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead II')
legend('DWT filtering','Location','Best')


%% Absolute value of amplitudes

y11 = cellfun(@abs, y1,'UniformOutput',false);
y22 = cellfun(@abs, y2,'UniformOutput',false);

%%
yy1 = mat2cell(cellfun(@(x)x.^2,y11,'UniformOutput',false),ones(1,1),ones(1,48));
yy2 = mat2cell(cellfun(@(x)x.^2,y22,'UniformOutput',false),ones(1,1),ones(1,48));

%% Findpeaks for both leads and all subjects
% Lead I
for i = 1:length(Data)
    [pks1{1,i},locs1{1,i}] = findpeaks(yy1{1,i}{1,1},tm1{1,i},'MinPeakHeight',...
    0.1,'MinPeakDistance',0.150);
end

% Lead II
for i = 1:length(Data)
    [pks2{1,i},locs2{1,i}] = findpeaks(yy2{1,i}{1,1},tm2{1,i},'MinPeakHeight',...
    0.1,'MinPeakDistance',0.150);
end
%% RR-interval

% Lead I
for i = 1:length(Data)
    RR_int1{1,i} = diff(locs1{1,i});
end

% Lead II
for i = 1:length(Data)
    RR_int2{1,i} = diff(locs2{1,i});
end

%plot(locs1{1,48},pks1{1,48},'ro')
%%

subplot(121)
plot(tm1{1,48},y11{1,48})
xlim([0 100])
subplot(122)
plot(tm2{1,48},y22{1,48})
xlim([0 100])


%%
subplot(121)
plot(tm{1,48},yy{1,48})
subplot(122)
plot(tm{1,48},yy{1,48})