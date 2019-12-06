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

%% Resample signal to fs/2

[p,q] = rat(125/250);

for i = 1:length(Data1)
    sig1_new{i} = resample(sig_1{i}, p, q);
    sig2_new{i} = resample(sig_2{i}, p, q);
end
clear sig_1 sig_2 tm_1 tm_2 p q Fs1 Fs2


%% Read annotations

for i = 1:length(Data1)
    [ann{i}, anntype{i}, subtype{i}, chan{i}, num{i}, comments{i}] = rdann(num2str(Data1(i),'%05.f'),'atr');
end

%% Find waves

for i = 1:length(Data1)
    %ecgpuwave(num2str(Data1(i),'%05.f'),'test');
    pwaves1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'p');
    twaves1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'t');
    QRS1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'N');
    [wave1{i},loc1{i}] = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'');
end

%%
tic
for i = 1:length(Data1)
    QRS1{1,i} = round(QRS1{1,i}./2);
   % ann{1,i} = round(ann{1,i}./2);
    wave1{1,i} = round(wave1{1,i}./2);
    pwaves1{1,i} = round(pwaves1{1,i}./2);
    twaves1{1,i} = round(twaves1{1,i}./2);
end
toc

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
tic
y1 = dwt1(sig1_new,8,Data1); % Denoising Lead I
y2 = dwt2(sig2_new,8,Data1); % Denoising Lead II

toc
%% Result of DWT filtering - plots
figure()
subplot(221)
plot(sig1_new{1,1})
title('Subject 1, lead I')
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(222)
plot(sig2_new{1,1})
title('Subject 1, lead II')
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(223)
plot(y1{1,1})
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, Lead I')
legend('DWT Filtering','Location','Best')

subplot(224)
plot(y2{1,1})
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, Lead II')
legend('DWT Filtering','Location','Best')

%% Segment the R peaks in 128 R peaks per segment
% Use the segmentation helper function
Data_128 = segmentation(QRS1);

%% RR-interval and HRV features
% Use the FeatureExtraction helper function
%[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg] = FeatureExtraction(QRS,Data_128,Data);

[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg,...
   m_RRIseg1, s_RRIseg1, r_RRIseg1, n_RRIseg1, CV_RRIseg1, minRRIseg1,...
   trainMatrix] = FeatureExtraction(QRS1,Data_128,Data1);


%% Constructing tensor
tic
tensor = TensorConstruct(y1,y2,Data_128);
toc

%% Wavelet + EMD
%clear y1 y2 sig_1 sig_2 tm_1 tm_2 patient_segments1 patient_segments2 patient1 ...
 %   patient2 loc ecg_segments1 ecg_segments2 Data_128

tic
max_wavelet_level = 8;
n = 5;

for i = 1:12
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
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/WDEC1','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/EMD1','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 8
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/WDEC2','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/EMD2','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 12
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/WDEC3','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/EMD3','EMD','-v7.3')
        clear EMD WDEC
    %elseif i == 16
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/WDEC4','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/EMD4','EMD','-v7.3')
        clear EMD WDEC
    %elseif i == 20
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/WDEC5','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/EMD5','EMD','-v7.3')
        clear EMD WDEC
    %elseif i == 23
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/WDEC6','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/AFDB-features/Downsampled/EMD6','EMD','-v7.3')
        clear EMD WDEC
    end
end

t = toc;
%%
