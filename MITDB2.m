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

%% Downsampling
[p,q] = rat(180/360);

for i = 1:length(Data)
    sig1_new{i} = resample(sig1{i}, p, q);
    sig2_new{i} = resample(sig2{i}, p, q);
end
clear sig1 sig2 tm1 tm2 p q Fs1 Fs2

%% Read annotations

for i = 1:length(Data)
    [ann{i}, anntype{i}] = rdann(num2str(Data(i)),'atr');
end
%% Segment annotations

ann_128 = segmentation(anntype);

%ANN = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(ann_128, 2), 'UniformOutput', false));


%% Plot of subject 48 - both leads
subplot(211)
plot(sig1_new{1,48})
title('Subject 48, lead I')
xlim([0 4000])
xlabel('Samples')
ylabel('Amplitude (mV)')

subplot(212)
plot(sig2_new{1,48})
title('Subject 48, lead II')
xlim([0 4000])
xlabel('Samples')
ylabel('Amplitude (mV)')

%% Denoising with DWT - Lead I and Lead II
 
y1 = dwt_denoise1(sig1_new,8,Data); % Denoising Lead I
y2 = dwt_denoise2(sig2_new,8,Data); % Denoising Lead II

%% Result of DWT filtering - plots
subplot(221)
plot(sig1_new{1,48})
title('Subject 48, lead I')
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(222)
plot(y1{1,48})
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 48, Lead I')
legend('DWT Filtering','Location','Best')

subplot(223)
plot(sig2_new{1,48})
title('Subject 48, lead II')
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')

subplot(224)
plot(y2{1,48})
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 48, Lead II')
legend('DWT Filtering','Location','Best')
%% Find R peaks and segment R peaks with 50 % overlap

for i = 1:length(Data)
    [qrs_amp_raw{i},qrs_i_raw{i},delay{i}] = pan_tompkin(y1{1,i},180,0);
end

Data_128 = segmentation(qrs_i_raw);


%% RR-interval and HRV features

[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg] = FeatureExtraction(qrs_i_raw,Data_128,Data);


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

%% Cutting the filtered signal in segments - Both leads, all patients
% Constructing tensor for each patient

[ecg_segments1, ecg_segments2, tensor] = TensorConstruct(y1,y2,Data_128);


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
plot(ecg_segments1{1,1}(1,18450:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead I, zeros')
subplot(212)
plot(ecg_segments2{1,1}(1,18450:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead II, zeros')


%% Wavelet + EMD
clear ecg_segments1 ecg_segments2 Data_128 y1 y2 sig1_new sig2_new
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
        save('/Volumes/TOSHIBA EXT/WDEC11','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/EMD11','EMD','-v7.3')
        clear EMD WDEC
    elseif i == 48
        save('/Volumes/TOSHIBA EXT/WDEC22','WDEC','-v7.3')
        save('/Volumes/TOSHIBA EXT/EMD22','EMD','-v7.3')
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

 
