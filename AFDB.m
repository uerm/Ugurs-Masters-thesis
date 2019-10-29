%% Load ECG signals
clear,clc

% Use '%05.f' to add leading zeros to number which is transformed to
% string by num2str.

Data1 = [04015, 04043, 04048, 04126, 04746, 04908, 04936, 05091, 05121, 05261 ...
    06426, 06453, 06995, 07162, 07859, 07879, 07910, 08215, 08219, 08378,...
    08405, 08434, 08455];


for i = 1:length(Data1)
    [sig_1{i}, Fs1, tm_1{i}] = rdsamp(num2str(Data1(i),'%05.f'),1); % Lead I
    [sig_2{i}, Fs2, tm_2{i}] = rdsamp(num2str(Data1(i),'%05.f'),2); % Lead II
end

%% Read annotations

for i = 1:length(Data1)
    [ann{i}, anntype{i}] = rdann(num2str(Data1(i),'%05.f'),'atr');
end

%% Find waves

for i = 1:length(Data1)
    ecgpuwave(num2str(Data1(i),'%05.f'),'test');
    pwaves1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'p');
    twaves1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'t');
    QRS1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'N');
    [wave1{i},loc1{i}] = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'');
end

%% Plot of subject 1 - both leads
subplot(211)
plot(tm_1{1,1},sig_1{1,1})
title('Subject 1, lead I')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')

subplot(212)
plot(tm_2{1,1},sig_2{1,1})
title('Subject 1, lead II')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')


%% Denoising with DWT - Lead I and Lead II
y1 = dwt_denoise1(sig_1,9,Data1); % Denoising Lead I
y2 = dwt_denoise2(sig_2,9,Data1); % Denoising Lead II

%% Result of DWT filtering - plots
figure()
subplot(221)
plot(tm_1{1,1},sig_1{1,1})
title('Subject 1, lead I')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(222)
plot(tm_1{1,1},sig_2{1,1})
title('Subject 1, lead II')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(223)
plot(tm_1{1,1},y1{1,1})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 1, Lead I')
legend('DWT Filtering','Location','Best')

subplot(224)
plot(tm_2{1,1},y2{1,1})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 1, Lead II')
legend('DWT Filtering','Location','Best')

%% Segmentation of RR intervals with 50% overlap
QRS_Length=length(QRS1);

Data_128 = zeros(0);

for i=1:QRS_Length
    L=1;
    K=1;
    m=0;
    for j=1:64:length(QRS1{i})
        if (L+128) > length(QRS1{i})
            continue;
        end
        Data_128{i}(K,:)=QRS1{i}((64*m+1):(L+127));
        L=L+64;
        K=K+1;
        m=m+1;
    end
end

%% RR-interval - with and without segmentation
RRI = zeros(0);

% RR-interval without segmentation.
for i = 1:length(Data1)
    RRI{1,i} = diff(QRS1{1,i}); 
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
minRRI = zeros(0);

for i = 1:length(Data1)
    M_RRI{1,i} = mean(RRI{1,i}); % Mean of RR intervals for each subject.
    SDNN{1,i} = std(RRI{1,i}); % Standard deviation of all RR intervals for each subject.
    RMSSD{1,i} = rms(diff(RRI{1,i})); % RMSSD of RR intervals diff for each subject.
    nRMSSD{1,i} = RMSSD{1,i}/M_RRI{1,i}; % nRMSSD of RR intervals.
    CV{1,i} = SDNN{1,i}/M_RRI{1,i}; % Coefficient of variation.
    minRRI{1,i} = min(RRI{1,i}); % Minimal RR interval.
end

% NN50
NN50 = zeros(0);
for i = 1:length(RRI) 
n = 0; 
    for num = 1:length(RRI{1,i})-1 
        if (RRI{1,i}(num+1)-RRI{1,i}(num) > 50*10^(-3)*250) 
        n = n+1; 
        end 
    end 
NN50{i} = n; 
end

% pNN50 (in percentage)
for i = 1:length(RRI)
    pNN50{1,i} = (NN50{1,i}/length(RRI{1,i}))*100;
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

%% Cutting the filtered signal in segments - Lead I, all patients
ecg_segments1 = {};
for p = 1:length(y1)
    patient1 = y1{1,p};
    peaks1 = Data_128{1,p};
    seg_lengths1 = peaks1(:,end)-peaks1(:,1)+1;
    max_len1 = max(seg_lengths1);
    patient_segments1 = zeros(0,max_len1);
    for i = 1:length(peaks1(:,1))
        seg1 = patient1(peaks1(i,1):peaks1(i,end));
        pad1 = zeros(1,max_len1-length(seg1));
        padded_seg1 = [seg1,pad1];
        patient_segments1(end+1,:) = padded_seg1;
    end
    ecg_segments1{end+1} = patient_segments1;
end

%% Cutting the filtered signal in segments - Lead II, all patients

ecg_segments2 = {};
for p = 1:length(y2)
    patient2 = y2{1,p};
    peaks2 = Data_128{1,p};
    seg_lengths2 = peaks2(:,end)-peaks2(:,1)+1;
    max_len2 = max(seg_lengths2);
    patient_segments2 = zeros(0,max_len2);
    for i = 1:length(peaks2(:,1))
        seg2 = patient2(peaks2(i,1):peaks2(i,end));
        pad2 = zeros(1,max_len2-length(seg2));
        padded_seg2 = [seg2,pad2];
        patient_segments2(end+1,:) = padded_seg2;
    end
    ecg_segments2{end+1} = patient_segments2;
end

%% Construct tensor for each patient

for i = 1:length(Data1)
    tensor{1,i} = cat(3,ecg_segments1{1,i},ecg_segments2{1,i});
end

%% Wavelet + EMD
clear y1 y2 sig_1 sig_2 tm_1 tm_2 patient_segments1 patient_segments2 patient1 ...
    patient2 loc ecg_segments1 ecg_segments2 Data_128
max_wavelet_level = 8;
n = 5;

for i = 1:length(Data1)
    i
    patient = tensor{1,i};
    for k = 1:size(patient,1)
        for j = 1:size(patient,3)
            WDEC{1,i}(k,:,:,j) = single(modwt(tensor{1,i}(k,:,j),max_wavelet_level,'db4'));
            for l =1:max_wavelet_level+1
                [imf,res] = emd(squeeze(WDEC{1,i}(k,l,:,j)),'Display',0);
                pad_size = max(0,n-size(imf,2));
                pad = zeros(size(imf,1),pad_size);
                padded_imf = cat(2,imf,pad);
                EMD{1,i}(k,l,:,:,j) = single(padded_imf(:,1:n));
            end
        end      
    end
    if i == 4
        save('/Volumes/TOSHIBA EXT/AFDB-features/WDEC1','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/EMD1','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 8
        save('/Volumes/TOSHIBA EXT/AFDB-features/WDEC2','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/EMD2','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 12
        save('/Volumes/TOSHIBA EXT/AFDB-features/WDEC3','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/EMD3','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 16
        save('/Volumes/TOSHIBA EXT/AFDB-features/WDEC4','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/EMD4','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 20
        save('/Volumes/TOSHIBA EXT/AFDB-features/WDEC5','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/EMD5','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 23
        save('/Volumes/TOSHIBA EXT/AFDB-features/WDEC6','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/EMD6','EMD','-v7.3')
        clear EMD WDEC
    end
end
%%
