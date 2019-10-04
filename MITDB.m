%% Load ECG signals
clear,clc

% MITDB Data
Data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,...
    114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203,...
    205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222,...
    223, 228, 230, 231, 232, 233, 234];


for i = 1:length(Data)
    [sig1{i}, Fs1, tm1{i}] = rdsamp(num2str(Data(i)),1); % Lead I
    [sig2{i}, Fs2, tm2{i}] = rdsamp(num2str(Data(i)),2); % Lead II
end

%% Read annotations

for i = 1:length(Data)
    [ann{i}, anntype{i}] = rdann(num2str(Data(i)),'atr');
end

%% Find waves and QRS complex

% ecgpuwave from the WFDB toolbox finds the onset, peak and offset of the
% P-QRS-T segment.
% pwaves = peak of p waves.
% twaves = peak of twaves.
% QRS = peak of QRS complex (R peak).
% wave = location of start, end and peak of waves.
% loc = notation of start, end and peak of waves.


for i = 1:length(Data)
    ecgpuwave(num2str(Data(i)),'test');
    pwaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'p');
    twaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'t');
    QRS{i} = rdann(num2str(Data(i)),'test',[],[],[],'N');
    [wave{i},loc{i}] = rdann(num2str(Data(i)),'test',[],[],[],'');
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

%% Denoising with DWT - Lead I and Lead II
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
plot(tm1{1,48}, y1{1,48})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead I')
legend('DWT filtering','Location','Best')

subplot(224)
plot(tm2{1,48}, y2{1,48})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead II')
legend('DWT filtering','Location','Best')
