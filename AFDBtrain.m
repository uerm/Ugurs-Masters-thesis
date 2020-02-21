%% Load ECG signals
clear,clc

% Use '%05.f' to add leading zeros to number which is transformed to
% string by num2str.

Data1 = [04015, 04043, 04048, 04126, 04746, 04908, 04936, 05091, 05121, 05261 ...
    06426, 06453, 06995, 07162, 07859, 07879];


for i = 1:length(Data1)
    [sig_1{i}, Fs1, ~] = rdsamp(num2str(Data1(i),'%05.f'),1); % Lead I
    [sig_2{i}, Fs2, ~] = rdsamp(num2str(Data1(i),'%05.f'),2); % Lead II
end

%% Denoise signal

y1 = dwt_denoise1(sig_1,8,Data1); % Denoising Lead I
y2 = dwt_denoise2(sig_2,8,Data1); % Denoising Lead II

%% Resample

[p,q] = rat(125/Fs1);

for i = 1:length(Data1)
    y1_new{i} = resample(y1{i}, p, q);
    y2_new{i} = resample(y2{i}, p, q);
end
clear p q

%% Read annotations

for i = 1:length(Data1)
    [ann2{i}, anntype2{i}, ~, ~, ~, comments{i}] = rdann(num2str(Data1(i),'%05.f'),'atr');
end

%% Find waves

for i = 1:length(Data1)
    ecgpuwave(num2str(Data1(i),'%05.f'),'test');
    pwaves1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'p');
    twaves1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'t');
    QRS1{i} = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'N');
    [wave1{i},loc1{i}] = rdann(num2str(Data1(i),'%05.f'),'test',[],[],[],'');
end

%%
tic
for i = 1:length(Data1)
    %QRS1{1,i} = round(QRS1{1,i}./2);
    ann2{1,i} = round(ann2{1,i}./2);
    %wave1{1,i} = round(wave1{1,i}./2);
    %pwaves1{1,i} = round(pwaves1{1,i}./2);
    %twaves1{1,i} = round(twaves1{1,i}./2);
end
toc




%%
for j = 1:length(Data1)
    signal_length = length(y11_new{j});
    anno_sig = zeros(1,signal_length);
    
    ann_ = [ann{j}; signal_length];
    
    for i=1:length(comments{j})
        if strcmp(comments{j}{i},'(AFIB')
            anno_sig(ann_(i):ann_(i+1)) = 1;
        end
    end
    all_ann{j} = anno_sig;
end

%% Cut the annotation according to the R peak indices

for i = 1:length(Data1)
    AL3{i} = all_ann2{i}(QRS2{i});
end
%% Segmentation of the labels

AL_128_1 = segmentation(AL3,20); % Define segment length

%M10 = threshold(AL_128,10); % Threshold 10%
%M20 = threshold(AL_128,20); % Threshold 20%
%M30 = threshold(AL_128,30); % Threshold 30%

% Use the "M" variables depending on the threshold that will be used.

M1 = cellfun(@(m)mode(m,2), AL_128_1,'uni',0); % Majority voting

% Stack the labels for classification
MM1 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M1, 2), 'UniformOutput', false));
%MM10 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M10, 2), 'UniformOutput', false));
%MM20 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M20, 2), 'UniformOutput', false));
%MM30 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M30, 2), 'UniformOutput', false));



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

%%
% y1_new = zeros(0);
% y2_new = zeros(0);
% 
% for i = 1:length(Data1)
%     y1_new{i} = y1{i};
%     y2_new{i} = y2{i};
% 
%     y1_new{i}(y1_new{i}> 1.5) = 1.5;
%     y2_new{i}(y2_new{i}> 1.5) = 1.5;
% end

%% Result of DWT filtering - plots
figure()
subplot(221)
plot(sig_1{1,1})
title('Subject 4015, Lead I')
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(222)
plot(sig_2{1,1})
title('Subject 4015, Lead II')
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(223)
plot(y1{1,1})
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 4015, Lead I')
legend('DWT Filtering','Location','Best')

subplot(224)
plot(y2{1,1})
xlim([0 3000])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 4015, Lead II')
legend('DWT Filtering','Location','Best')

%% Segment the R peaks in 128 R peaks per segment
% Use the segmentation helper function
Data_128 = segmentation(QRS2,20);

%% RR-interval and HRV features
% Use the FeatureExtraction helper function
%[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg] = FeatureExtraction(QRS,Data_128,Data);

Fs = 125;

[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~,trainMatrix]...
    = FeatureExtraction(QRS2,Data_128,Data1,Fs);


%% Constructing tensor
tic
[ecg_segments1, ecg_segments2, ~] = TensorConstruct(y1_new,y2_new,Data_128);
toc

ecg1 = cellfun(@(m) normalize(m,2), ecg_segments1,'uni',0);
ecg2 = cellfun(@(m) normalize(m,2), ecg_segments2,'uni',0);

for i = 1:length(Data1)
    tensor{i} = cat(3,ecg1{i},ecg2{i});
end


%% Wavelet + EMD

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
end



%% Permute EMDs
tic
% Permute all EMD arrays
for i = 1:length(EMD)
    perm{1,i} = permute(EMD{1,i},[2,4,5,1,3]);
end
toc
%% Reshape EMDs
tic 
% Reshape all the perm matrices
for i = 1:length(perm)
    resh{1,i} = reshape(perm{1,i},[],size(perm{1,i},5),size(perm{1,i},4));
end
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

% Because of the large memory requirements, this part of the algorithm is
% computed using the DTU High Performance Computers.
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

skew = squeeze(skew);
kurt = squeeze(kurt);

feat_cat = cat(1,mu,st,v,skew,kurt);

%% -------------------------- Classification ----------------------------
load('feat_cat1')
load('feat_cat2')
load('feat_cat3')
load('feat_cat4')
load('feat_cat5')
load('feat_cat6')

feat_cat = cat(2,feat_cat1,feat_cat2,feat_cat3,feat_cat4);
feat_cat = feat_cat';

trainMatrix128 = [feat_cat trainMatrix MM1];

%trainMatrix128_30 = [feat_cat trainMatrix MM30];
%trainMatrix128_20 = [feat_cat trainMatrix MM20];
%trainMatrix128_10 = [feat_cat trainMatrix MM10];

%% Gather the different parts into one matrix, M = 60
load('feat_cat1')
load('feat_cat2')
load('feat_cat3')
load('feat_cat4')


feat_cat = cat(2,feat_cat1,feat_cat2,feat_cat3,feat_cat4);
feat_cat = feat_cat';

trainMatrix60 = [feat_cat trainMatrix MM1];
%trainMatrix60_30 = [feat_cat trainMatrix MM30];
%trainMatrix60_20 = [feat_cat trainMatrix MM20];
%trainMatrix60_10 = [feat_cat trainMatrix MM10];

%% M = 20
load('feat_cat1')
load('feat_cat21')
load('feat_cat22')
load('feat_cat3')
load('feat_cat4')


feat_cat = cat(2,feat_cat1,feat_cat21,feat_cat22,feat_cat3,feat_cat4);
feat_cat = feat_cat';

trainMatrix20 = [feat_cat trainMatrix MM1];



%% Classify data with trained models

x = trainMatrix20(:,452);
y = trainMatrix20(:,451);
scatterhist(x,y,'Group',trainMatrix20(:,end),'Kernel','on','Location','SouthEast',...
    'Direction','out','Color','br')
xlabel('Standard Deviation of RRi (F452)')
ylabel('Mean RRi (F451)')
title('AFDB, majority vote, M = 128 beats')


%% 

[preds,scores] = predict(classification3KNN128ADASYN,trainMatrix128(:,1:end-1));
% 
% [x,y,~,auc] = perfcurve(trainMatrix128(:,end),scores(:,2),1);



% figure()
% plot(x,y,'LineWidth',2)
% xlabel('False Positive Rate')
% ylabel('True Positive Rate')
% title('ROC, AFDB, 50 Trees, Test')

figure()
confusionchart(trainM(:,end),yfit);
title('AFDB, 50 Trees, Test')



