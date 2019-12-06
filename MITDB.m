%% Load ECG signals
clear,clc

addpath '/Users/ugurerman95/Documents/mcode' % Add path to WFDB toolbox

% MITDB Data
Data = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 111, 112, 113,...
    114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 200, 201, 202, 203,...
    205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222,...
    223, 228, 230, 231, 232, 233, 234];


for i = 1:length(Data)
    [sig1{i}, Fs1, tm1{i}] = rdsamp(num2str(Data(i)),1); % Lead I
    [sig2{i}, Fs2, tm2{i}] = rdsamp(num2str(Data(i)),2); % Lead II
end

%% Read annotations and segment annotation

for i = 1:length(Data)
    [ann{i}, anntype{i}, subtype{i}, chan{i}, num{i}, comments{i}] = rdann(num2str(Data(i)),'atr');
end

ann_Length=length(anntype);

ann_128 = zeros(0);

for i=1:ann_Length
    L=1;
    K=1;
    m=0;
    for j=1:64:length(anntype{i})
        if (L+128) > length(anntype{i})
            continue;
        end
        ann_128{i}(K,:)=anntype{i}((64*m+1):(L+127));
        L=L+64;
        K=K+1;
        m=m+1;
    end
end

ANN = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(ann_128, 2), 'UniformOutput', false));

%% Find waves and QRS complex

% ecgpuwave from the WFDB toolbox finds the onset, peak and offset of the
% P-QRS-T segment.
% pwaves = peak of p waves.
% twaves = peak of twaves.
% QRS = peak of QRS complex (R peak).
% wave = location of start, end and peak of waves.
% loc = notation of start, end and peak of waves.


for i = 1:length(Data)
    %ecgpuwave(num2str(Data(i)),'test');
    pwaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'p');
    twaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'t');
    QRS{i} = rdann(num2str(Data(i)),'test',[],[],[],'N');
    [wave{i},loc{i}] = rdann(num2str(Data(i)),'test',[],[],[],'');
end


%% Plot of subject 48 - both leads
subplot(211)
plot(tm1{1,48}, sig1{1,48})
title('Subject 48, lead I')
xlim([0 40])
xlabel('Samples')
ylabel('Amplitude (mV)')

subplot(212)
plot(tm1{1,48}, sig2{1,48})
title('Subject 48, lead II')
xlim([0 40])
xlabel('Samples')
ylabel('Amplitude (mV)')

%% Denoising with DWT - Lead I and Lead II
y1 = dwt_denoise1(sig1,9,Data); % Denoising Lead I
y2 = dwt_denoise2(sig2,9,Data); % Denoising Lead II

%% Result of DWT filtering - plots
subplot(221)
plot(tm1{1,48},sig1{1,48})
title('Subject 48, lead I')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(222)
plot(tm1{1,48},y1{1,48})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead I')
legend('DWT Filtering','Location','Best')

subplot(223)
plot(tm1{1,48},sig2{1,48})
title('Subject 48, lead II')
xlim([0 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')

subplot(224)
plot(tm2{1,48},y2{1,48})
xlim([1 40])
xlabel('Time (s)')
ylabel('Amplitude (mV)')
title('Subject 48, Lead II')
legend('DWT Filtering','Location','Best')

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
minRRI = zeros(0);

for i = 1:length(Data)
    M_RRI{1,i} = mean(RRI{1,i}); % Mean of RR intervals for each subject.
    SDNN{1,i} = std(RRI{1,i}); % Standard deviation of all RR intervals for each subject.
    RMSSD{1,i} = rms(diff(RRI{1,i})); % RMSSD of RR intervals diff for each subject.
    nRMSSD{1,i} = RMSSD{1,i}/M_RRI{1,i}; % nRMSSD of RR intervals.
    CV{1,i} = SDNN{1,i}/M_RRI{1,i}; % Coefficient of variation.
    minRRI{1,i} = min(RRI{1,i}); % Minimal RR interval.
end

% NN50
for i = 1:length(RRI)
    n = 0;
    for num = 1:length(RRI{1,i})-1
        if (RRI{1,i}(num+1)-RRI{1,i}(num) > 50*10^(-3)*360)
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

m_RRIseg1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(m_RRIseg, 2), 'UniformOutput', false));

% Standard deviation of RR intervals (of each row)
s_RRIseg = cellfun(@(m)std(m,0,2),RRIseg,'uni',0);

s_RRIseg1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(s_RRIseg, 2), 'UniformOutput', false));

% RMSSD of RR interal segments
r_RRIseg = cellfun(@(m)rms(diff(m,1,2),2),RRIseg,'uni',0);

r_RRIseg1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(r_RRIseg, 2), 'UniformOutput', false));

% nRMSSD of RR interval segments
n_RRIseg = cellfun(@(x,y) x./y, r_RRIseg, m_RRIseg, 'uni',0);

n_RRIseg1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(n_RRIseg, 2), 'UniformOutput', false));

% Coefficient of variation of RR segments
CV_RRIseg = cellfun(@(x,y) x./y, s_RRIseg, m_RRIseg, 'uni',0);

CV_RRIseg1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(CV_RRIseg, 2), 'UniformOutput', false));

% Minimal RR interval of segments.
minRRIseg = cellfun(@(m) min(m,[],2), RRIseg,'uni',0);

minRRIseg1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(minRRIseg, 2), 'UniformOutput', false));

trainMatrix = [m_RRIseg1, s_RRIseg1, r_RRIseg1, n_RRIseg1, CV_RRIseg1, minRRIseg1];


%% Segmentation of filtered signal based on R peak location
subplot(211)
plot(y1{1,1}(1:Data_128{1}(1,end)))
title('Segmentation (first 128 R peaks), subject 1, lead I, no zero padding')
xlabel('Samples')
ylabel('Amplitude (mV)')
subplot(212)
plot(y2{1,1}(1:Data_128{1}(1,end)))
title('Segmentation (first 128 R peaks), subject 1, lead II, no zero padding')
xlabel('Samples')
ylabel('Amplitude (mV)')

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

%% Plot of ECG with zero padding
figure()
subplot(211)
plot(ecg_segments1{1,1}(1,:))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Segmentation (first 128 R peaks), subject 1, lead I, with zero padding')
subplot(212)
plot(ecg_segments2{1,1}(1,:))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Segmentation (first 128 R peaks), subject 1, lead II, with zero padding')

figure()
subplot(211)
plot(ecg_segments1{1,1}(1,37000:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead I, zeros')
subplot(212)
plot(ecg_segments2{1,1}(1,37000:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead II, zeros')


%% Construct tensor for each patient

for i = 1:length(Data)
    tensor{1,i} = cat(3,ecg_segments1{1,i},ecg_segments2{1,i});
end

%% Wavelet + EMD
clear sig1 sig2 tm1 tm2 patient_segments1 patient_segments2 patient1 ...
    patient2 loc ecg_segments1 ecg_segments2 Data_128
max_wavelet_level = 8;
n = 5;

for i = 1:length(Data)
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
    if i == 24
        save('/Volumes/TOSHIBA EXT/WDEC1','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/EMD1','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 48
        save('/Volumes/TOSHIBA EXT/WDEC2','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/EMD2','EMD','-v7.3')
        clear EMD WDEC
    end
end

%%

for i = 1:24
    EMD2{1,i} = EMD1{1,i};
end






%% Permute EMDs

% Permute all EMD arrays
for i = 1:length(EMD2)
    perm{1,i} = permute(EMD2{1,i},[2,4,5,1,3]);
end

%% Reshape EMDs

% Reshape all the perm matrices
for i = 1:length(EMD2)
    resh{1,i} = reshape(perm{1,i},[],size(perm{1,i},5),size(perm{1,i},4));
end

%% Zero padding the tensor to max length
tic
[~,b,~] = cellfun(@size, resh);

idx_pad = max(b) - b;

for idx = 1:length(resh)
    idx
    resh_padded{1,idx} = padarray(resh{1,idx}, [0 idx_pad(idx) 0],0,'post');
    
    if idx == 24
        save('/Volumes/TOSHIBA EXT/MITDB-features/resh_padded1','resh_padded','-v7.3')
        clear resh_padded
    elseif idx == 48
        save('/Volumes/TOSHIBA EXT/MITDB-features/resh_padded2','resh_padded','-v7.3')
        clear resh_padded
    end
end
t = toc;

%% Train autoencoder
hidden_size = 500;

%for i = 1:length(EMD)
    acode1{1} = trainAutoencoder(squeeze(resh{1}(11,:,:)),hidden_size);
%end

%save('/Volumes/TOSHIBA EXT/acode2','acode2','-v7.3')

%% Test to see which DWT or EMD should be discarded
for i=11
    test_signal = [];
    test_signal(:,1) = resh{1}(i,:,1);
    
    size(test_signal);
    %pred = predict(acode1{1,i},test_signal);
    
    figure()
    %plot(pred)
    %hold on
    plot(test_signal)
    %hold off
end

%% Mean squared error - n = 50 and n = 100
for i = 1:24
    Xrec1{1,i} = predict(acode1{1},squeeze(resh{1,1}(11,:,1))');
    ms1{1,i} = mse(squeeze(resh{1,1}(11,:,1)) - Xrec1{1,i}');
    %Xrec2{1,i} = predict(acode2{1,i},squeeze(resh{1,i}(1,:,:)));
    %ms2{1,i} = mse(squeeze(resh{1,i}(1,:,:)) - Xrec2{1,i});
end

%mss = [ms;ms1];

%%



figure()
plot(resh{1,1}(11,:,1),'r');
hold on
plot(Xrec1{1,1},'g');
%enc = ['fig',num2str(i),'.png'];
%saveas(cgca,enc);

 
