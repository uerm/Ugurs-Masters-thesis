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
y1 = dwt_denoise1(sig1,9,Data); % Denoising Lead I
y2 = dwt_denoise2(sig2,9,Data); % Denoising Lead II

%% Result of DWT filtering - plots

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

%% Segmentation of RR intervals with 50% overlap
QRS_Length=length(QRS);

Data_128 = zeros(0);

for i=1:QRS_Length
    L=1;
    K=1;
    m=0;
    for j=1:64:length(QRS{i})
        if (L+128) > length(QRS{i})
            continue;
        end
        Data_128{i}(K,:)=QRS{i}((64*m+1):(L+127));
        L=L+64;
        K=K+1;
        m=m+1;
    end
end


%% RR-interval - with and without segmentation
RRI = zeros(0);

% RR-interval without segmentation.
for i = 1:length(Data)
    RRI{1,i} = diff(QRS{1,i}); 
end

% RR-interval with segmentation.
%fun = @(m)diff(m,1,2);
RRIseg = cellfun(@(m)diff(m,1,2),Data_128,'uni',0);



%% Calculate HRV features - without segmentation
M_RRI = zeros(0);
SDNN = zeros(0);
RMSSD = zeros(0);
nRMSSD = zeros(0);
NN50 = zeros(0);
pNN50 = zeros(0);
CV = zeros(0);
minRR = zeros(0);

for i = 1:length(Data)
    M_RRI{1,i} = mean(RRI{1,i}); % Mean of RR intervals for each subject.
    SDNN{1,i} = std(RRI{1,i}); % Standard deviation of all RR intervals for each subject.
    RMSSD{1,i} = rms(diff(RRI{1,i})); % RMSSD of RR intervals diff for each subject.
    nRMSSD{1,i} = RMSSD{1,i}/M_RRI{1,i}; % nRMSSD of RR intervals.
    CV{1,i} = SDNN{1,i}/M_RRI{1,i}; % Coefficient of variation.
    minRRI{1,i} = min(RRI{1,i}); % Minimal RR interval.
end
%% NN50
m = 0;
for num = 1:length(RRI{1,3})-1
    if (RRI{1,3}(num+1)-RRI{1,3}(num) > 50*10^(-3)*360)
        m = m+1;
    end
end


%% Calculate HRV features - with segmentation

% Mean of the RR intervals (of each row)
m_RRIseg = cellfun(@(m)mean(m,2),RRIseg,'uni',0);

% Standard deviation of RR intervals (of each row)
s_RRIseg = cellfun(@(m)std(m,0,2),RRIseg,'uni',0);

% RMSSD of RR interal segments
r_RRIseg = cellfun(@(m)rms(diff(m,1,2),2),RRIseg,'uni',0);

% nRMSSD of RR interval segments
n_RRIseg = cellfun(@(x,y) x./y, r_RRIseg, m_RRIseg, 'uni',0);

% Coefficient of variation of RR segments
CV_RRIseg = cellfun(@(x,y) x./y, s_RRIseg, m_RRIseg, 'uni',0);

% Minimal RR interval of segments.
minRRIseg = cellfun(@(m) min(m,[],2), RRIseg,'uni',0);


%% Segmentation of filtered signal based on R peak location
subplot(211)
plot(y1{1,1}(1:Data_128{1}(1,end)))
title('Segmentation (first 128 R peaks), subject 1, lead I')
xlabel('Samples')
ylabel('Amplitude (mV)')
subplot(212)
plot(y2{1,1}(1:Data_128{1}(1,end)))
title('Segmentation (first 128 R peaks), subject 1, lead II')
xlabel('Samples')
ylabel('Amplitude (mV)')

y1_seg{1,1} = y1{1,1}(1:Data_128{1}(1,end));

y1_seg = zeros(0);

for i = 1:34
    y1_seg{1,i} = y1{1,1}(Data_128{1}(i,i):Data_128{1}(i,end));
end



%%
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

%%
cnt = 1;
ecg_segments = {};
for p = 1:length(y1)
    patient = y1{1,p};
    peaks = Data_128{1,p};
    seg_lengths = peaks(:,end)-peaks(:,1)+1;
    max_len = max(seg_lengths);
    patient_segments = zeros(0,max_len);
    for i = 1:length(peaks(:,1))
        seg = patient(peaks(i,1):peaks(i,end));
        pad = zeros(1,max_len-length(seg));
        padded_seg = [seg,pad];
        patient_segments(end+1,:) = padded_seg;
    end
    ecg_segments{end+1} = patient_segments;
end
