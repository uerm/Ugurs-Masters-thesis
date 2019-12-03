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

% for i = 1:length(Data)
%     [qrs_amp_raw{i},qrs_i_raw{i},delay{i}] = pan_tompkin(y1{1,i},180,0);
% end

for i = 1:length(Data)
    %ecgpuwave(num2str(Data(i)),'test');
    pwaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'p');
    twaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'t');
    QRS{i} = rdann(num2str(Data(i)),'test',[],[],[],'N');
    [wave{i},loc{i}] = rdann(num2str(Data(i)),'test',[],[],[],'');
end


%% Correcting R peak indeces
% Divide each peak index with 2 since resampling to fs/2
tic
for i = 1:length(Data)
    QRS{1,i} = round(QRS{1,i}./2);
    ann{1,i} = round(ann{1,i}./2);
    wave{1,i} = round(wave{1,i}./2);
    pwaves{1,i} = round(pwaves{1,i}./2);
    twaves{1,i} = round(twaves{1,i}./2);
end
toc
%%
tic
anntype_new = annotation2(Data,loc,wave,ann,anntype);

AL = labelling(anntype_new,Data);

AL_128 = segmentation(AL);

M = cellfun(@(m)mode(m,2), AL_128,'uni',0);

MM = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M, 2), 'UniformOutput', false));

toc
%% Segment the R peaks in 128 R peaks per segment
% Use the segmentation helper function
Data_128 = segmentation(QRS);

%% RR-interval and HRV features
% Use the FeatureExtraction helper function
%[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg] = FeatureExtraction(QRS,Data_128,Data);

[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg,...
   m_RRIseg1, s_RRIseg1, r_RRIseg1, n_RRIseg1, CV_RRIseg1, minRRIseg1,...
   trainMatrix] = FeatureExtraction(QRS,Data_128,Data);


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
plot(ecg_segments1{1,1}(1,18480:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead I, zeros')
subplot(212)
plot(ecg_segments2{1,1}(1,18480:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead II, zeros')


%% Wavelet + EMD
tic
clear ecg_segments1 ecg_segments2 Data_128 y1 y2 sig1_new sig2_new ...
    twaves loc wave pwaves
max_wavelet_level = 8;
nn = 5;

for i = 1:length(Data)
    i
    patient = tensor{1,i};
    for k = 1:size(patient,1)
        for j = 1:size(patient,3)
            WDEC{1,i}(k,:,:,j) = single(modwt(tensor{1,i}(k,:,j),max_wavelet_level,'db4'));
            for l =1:max_wavelet_level+1
                [imf,res] = emd(squeeze(WDEC{1,i}(k,l,:,j)),'Display',0);
                pad_size = max(0,nn-size(imf,2));
                pad = zeros(size(imf,1),pad_size);
                padded_imf = cat(2,imf,pad);
                EMD{1,i}(k,l,:,:,j) = single(padded_imf(:,1:nn));
            end
        end
    end
end
t = toc;

%% Permute EMDs

% Permute all EMD arrays
for i = 1:length(EMD)
    perm{1,i} = permute(EMD{1,i},[2,4,5,1,3]);
end

%% Reshape EMDs

% Reshape all the perm matrices
for i = 1:length(perm)
    resh{1,i} = reshape(perm{1,i},[],size(perm{1,i},5),size(perm{1,i},4));
end

%% Zero padding the tensor to max length
tic
[~,b,~] = cellfun(@size, resh);

idx_pad = max(b) - b;

for index = 1:length(resh)
    resh_padded{1,index} = padarray(resh{1,index}, [0 idx_pad(index) 0],0,'post');    
end
t = toc;

%% Concatenate zeropadded 3D arrays along 3rd axis
stacked = cat(3,resh_padded{:});

%% Calculate statistical features of DWT+EMD

mu = double(mean(stacked,2)); % mean
st = double(std(stacked,0,2)); % standard deviation
v = double(var(stacked,0,2)); % variance
mest = double(max(stacked,[],2)); % maximum
mindst = double(min(stacked,[],2)); % minimum

med = nan(90,1,1633); % median
skew = nan(90,1,1633); % skewness
kurt = nan(90,1,1633); % kurtosis

for i = 1:size(med,1)
    med(i,:,:) = double(median(stacked(i,:,:),2));
    skew(i,:,:) = double(skewness(stacked(i,:,:),1,2));
    kurt(i,:,:) = double(kurtosis(stacked(i,:,:),1,2));
end
%%
feat_cat = cat(2,mu,st,v,skew,kurt);

%% Train with only HRV features

trainMatrix1 = [trainMatrix MM];

%% Train with only DWT+EMD statistical features

feat_cat = feat_cat';

trainMatrix2 = [feat_cat MM];

%% Train with HRV features + DWT+EMD features

trainMatrix3 = [feat_cat trainMatrix MM];