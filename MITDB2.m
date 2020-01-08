%% Load ECG signals
clear,clc
addpath 'C:\Users\Dell\Desktop\Test\mit-bih-arrhythmia-database-1.0.0' % Add path to WFDB toolbox

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
    [ann{i},anntype{i},subtype{i},chan{i},num{i},comments{i}] = rdann(num2str(Data(i)),'atr');
end

comments2 = comments;
for i = 1:length(Data)
    for j = 2:length(comments2{1,i})
        if isempty(comments2{1,i}{j,:})
            comments2{1,i}{j,:} = comments2{1,i}{j-1,:};
        end
        
    end
end

%% Find R peaks and segment R peaks with 50 % overlap

tic
for i = 1:length(Data)
    %ecgpuwave(num2str(Data(i)),'test');
    pwaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'p');
    twaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'t');
    QRS{i} = rdann(num2str(Data(i)),'test',[],[],[],'N');
    [wave{i},loc{i}] = rdann(num2str(Data(i)),'test',[],[],[],'');
end
toc

%% Correcting R peak indeces
% Divide each peak index with 2 since resampling to fs/2
tic
for i = 1:length(Data)
    QRS{1,i} = round(QRS{1,i}./2);
    ann{1,i} = round(ann{1,i}./2);
    wave{1,i} = round(wave{1,i}./2);
end
toc
%%
tic

% "Cut" the labels according to the R peaks
anntype_new = annotation2(Data,loc,wave,ann,comments2);

AL = labelling2(anntype_new,Data); % Label the data

AL_128 = segmentation(AL,10); % Define segment length

M = threshold(AL_128,10); % Threshold 30%

%M = cellfun(@(m)mode(m,2), AL_128,'uni',0); % Majority voting

% Stack the labels for classification
MM = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M, 2), 'UniformOutput', false));

toc


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



%% Segment the R peaks in 128 R peaks per segment
% Use the segmentation helper function
Data_10 = segmentation(QRS,10);

%% RR-interval and HRV features
% Use the FeatureExtraction helper function

[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg,...
   m_RRIseg1, s_RRIseg1, r_RRIseg1, n_RRIseg1, CV_RRIseg1, minRRIseg1,...
   trainMatrix] = FeatureExtraction(QRS,Data_10,Data);


%% Segmentation of filtered signal based on R peak location
subplot(211)
plot(y1{1,1}(1:Data_10{1}(1,end)))
title('Segmentation (first 10 R peaks), subject 1, lead I, no zero padding')
xlabel('Samples')
ylabel('Amplitude (mV)')
subplot(212)
plot(y2{1,1}(1:Data_10{1}(1,end)))
title('Segmentation (first 10 R peaks), subject 1, lead II, no zero padding')
xlabel('Samples')
ylabel('Amplitude (mV)')

%% Cutting the filtered signal in segments - Both leads, all patients
% Constructing tensor for each patient

[ecg_segments1, ecg_segments2, tensor] = TensorConstruct(y1,y2,Data_10);

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
plot(ecg_segments1{1,1}(1,1200:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead I, zeros')
subplot(212)
plot(ecg_segments2{1,1}(1,1200:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 1, lead II, zeros')


%% Wavelet + EMD
tic
clear ecg_segments1 ecg_segments2 Data_10 y1 y2 sig1_new sig2_new ...
    twaves pwaves
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
toc

%% Permute EMDs
tic
% Permute all EMD arrays
for i = 1:length(EMD)
    perm{1,i} = permute(EMD{1,i},[2,4,5,1,3]);
end
clear EMD
toc
%% Reshape EMDs
tic 
% Reshape all the perm matrices
for i = 1:length(perm)
    resh{1,i} = reshape(perm{1,i},[],size(perm{1,i},5),size(perm{1,i},4));
end
clear perm
toc
%% Zero padding the tensor to max length
tic
[~,b,~] = cellfun(@size, resh);

idx_pad = max(b) - b;

for index = 1:length(resh)
    index
    resh_padded{1,index} = padarray(resh{1,index}, [0 idx_pad(index) 0],0,'post');    
end
t = toc;

%% Concatenate zeropadded 3D arrays along 3rd axis
clear resh
stacked = cat(3,resh_padded{:});
clear resh_padded

%% Calculate statistical features of DWT+EMD
tic
mu = squeeze(double(mean(stacked,2))); % mean
st = squeeze(double(std(stacked,0,2))); % standard deviation
v = squeeze(double(var(stacked,0,2))); % variance

skew = nan(90,1,size(stacked,3)); % skewness
kurt = nan(90,1,size(stacked,3)); % kurtosis

for i = 1:size(skew,1)
    skew(i,:,:) = double(skewness(stacked(i,:,:),1,2));
    kurt(i,:,:) = double(kurtosis(stacked(i,:,:),1,2));
end
toc

med = squeeze(med);
skew = squeeze(skew);
kurt = squeeze(kurt);

%%
feat_cat = cat(1,mu,st,v,skew,kurt);

%% Train with only HRV features (also with ADASYN)

trainMatrix1 = [trainMatrix MM];
% trainMatrix11 = trainMatrix1;
% indices1 = find(trainMatrix11(:,end)==2);
% trainMatrix11(indices1,:) = [];

feat1 = trainMatrix1(:,1:end-1);

MM1 = trainMatrix1(:,end);

[out_featuresSyn1, out_labelsSyn1] = ADASYN(feat1, MM1, 1,...
    5, 5, true);

t1 = [out_featuresSyn1 double(out_labelsSyn1)];

trainMatrix11 = [trainMatrix1;t1];


%% Train with only DWT+EMD statistical features (also with ADASYN)

feat_cat = feat_cat';

trainMatrix2 = [feat_cat MM];
%trainMatrix22= trainMatrix2;
%indices2 = find(trainMatrix22(:,end)==2);
%trainMatrix22(indices2,:) = [];


feat2 = trainMatrix2(:,1:end-1);

MM2 = trainMatrix2(:,end);

[out_featuresSyn2, out_labelsSyn2] = ADASYN(feat2, MM2, 1,...
    5, 5, true);

t2 = [out_featuresSyn2 double(out_labelsSyn2)];

trainMatrix22 = [trainMatrix2;t2];

%% Train with HRV features + DWT+EMD features (also with ADASYN)

trainMatrix3 = [feat_cat trainMatrix MM];
%trainMatrix4 = trainMatrix3;
%indices3 = find(trainMatrix4(:,end)==2);
%trainMatrix4(indices3,:) = [];

feat3 = trainMatrix3(:,1:end-1);

MM3 = trainMatrix3(:,end);

[out_featuresSyn3, out_labelsSyn3] = ADASYN(feat3, MM3, 1,...
    5, 5, true);

t3 = [out_featuresSyn3 double(out_labelsSyn3)];

trainMatrix33 = [trainMatrix3;t3];