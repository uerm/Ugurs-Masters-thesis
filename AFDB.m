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
    [ann{i}, ~, ~, ~, ~, comments{i}] = rdann(num2str(Data1(i),'%05.f'),'atr');
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
    ann{1,i} = round(ann{1,i}./2);
    %wave1{1,i} = round(wave1{1,i}./2);
    %pwaves1{1,i} = round(pwaves1{1,i}./2);
    %twaves1{1,i} = round(twaves1{1,i}./2);
end
toc

%%
for j = 1:length(Data1)
    signal_length = length(sig1_new{j});
    anno_sig = zeros(1,signal_length);
    
    ann_ = [ann{j}; signal_length];
    
    for i=1:length(comments{j})
        if strcmp(comments{j}{i},'(AFIB')
            anno_sig(ann_(i):ann_(i+1)) = 1;
        end
    end
    all_ann{j} = anno_sig;
end

%% Cut the annotation according to the R peak indeces

for i = 1:length(Data1)
    AL2{i} = all_ann{i}(QRS1{i});
end
%% Segmentation of the labels

AL_128 = segmentation(AL2,128); % Define segment length

%M10 = threshold(AL_128,10); % Threshold 10%
%M20 = threshold(AL_128,20); % Threshold 20%
%M30 = threshold(AL_128,30); % Threshold 30%

% Use the "M" variables depending on the threshold that will be used.

M2 = cellfun(@(m)mode(m,2), AL_128,'uni',0); % Majority voting

% Stack the labels for classification
MM2 = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M2, 2), 'UniformOutput', false));
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


%% Denoising with DWT - Lead I and Lead II
tic
y1 = dwt1(sig1_new,8,Data1); % Denoising Lead I
y2 = dwt2(sig2_new,8,Data1); % Denoising Lead II

toc
%%
y1_new = zeros(0);
y2_new = zeros(0);

for i = 1:length(Data1)
    y1_new{i} = y1{i};
    y2_new{i} = y2{i};

    y1_new{i}(y1_new{i}> 1.5) = 1.5;
    y2_new{i}(y2_new{i}> 1.5) = 1.5;
end

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
Data_128 = segmentation(QRS1,128);

%% RR-interval and HRV features
% Use the FeatureExtraction helper function
%[m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg] = FeatureExtraction(QRS,Data_128,Data);

[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~,trainMatrixAF]...
    = FeatureExtraction(QRS1,Data_128,Data1);


%% Constructing tensor
tic
[~, ~, tensor] = TensorConstruct(y1_new,y2_new,Data_128);
toc

%% Wavelet + EMD
%clear y1 y2 sig_1 sig_2 tm_1 tm_2 patient_segments1 patient_segments2 patient1 ...
 %   patient2 loc ecg_segments1 ecg_segments2 Data_128

tic
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

t = toc;


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

%% Gather the different parts into one matrix, M = 128
load('feat_cat1')
load('feat_cat2')
load('feat_cat3')
load('feat_cat4')
load('feat_cat5')
load('feat_cat6')

feat_cat = cat(2,feat_cat1,feat_cat2,feat_cat3,feat_cat4,feat_cat5,feat_cat6);
feat_cat = feat_cat';

trainMatrix128 = [feat_cat trainMatrixAF];
%trainMatrix128_30 = [feat_cat trainMatrix MM30];
%trainMatrix128_20 = [feat_cat trainMatrix MM20];
%trainMatrix128_10 = [feat_cat trainMatrix MM10];

%% Gather the different parts into one matrix, M = 60
load('feat_cat1')
load('feat_cat2')
load('feat_cat3')
load('feat_cat4')
load('feat_cat5')
load('feat_cat6')

feat_cat = cat(2,feat_cat1,feat_cat2,feat_cat3,feat_cat4,feat_cat5,feat_cat6);
feat_cat = feat_cat';

trainMatrix60 = [feat_cat trainMatrix MM];
trainMatrix60_30 = [feat_cat trainMatrix MM30];
trainMatrix60_20 = [feat_cat trainMatrix MM20];
trainMatrix60_10 = [feat_cat trainMatrix MM10];

%% Classify data with trained models

% Majority vote
tic
[predsCubicSVM128,scoresCubicSVM128] = predict(classificationCubicSVM128ADASYN,trainMatrix3);
t1 = toc;


% Threshold 30
tic
[predsCubicSVM128_30,scoresCubicSVM128_30] = predict(classificationCubicSVM128ADASYN_30,trainMatrix3);
t2 = toc;


% Threshold 20
tic
[predsCubicSVM128_20,scoresCubicSVM128_20] = predict(classificationCubicSVM128ADASYN_20,trainMatrix3);
t3 = toc;


% Threshold 10
tic
[predsCubicSVM128_10,scoresCubicSVM128_10] = predict(classificationCubicSVM128ADASYN_10,trainMatrix3);
t4 = toc;


% Majority vote
[xCubicSVM128,yCubicSVM128,~,aucCubicSVM128] = perfcurve(MM,scoresCubicSVM128(:,2),1);


figure()
plot(xCubicSVM128,yCubicSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM, AFDB, Test')

figure()
confusionchart(MM,predsCubicSVM128);
title('Cubic SVM, AFDB, Test')



%% Test
tic
[predsCubicSVM60,scoresCubicSVM60] = predict(classificationCubicSVM60ADASYN,trainMatrix60(:,1:end-1));
t1 = toc;


% Majority vote
[xCubicSVM60,yCubicSVM60,~,aucCubicSVM60] = perfcurve(MM,scoresCubicSVM60(:,2),1);


figure()
plot(xCubicSVM60,yCubicSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM, AFDB, Test')

figure()
confusionchart(MM,predsCubicSVM60);
title('Cubic SVM, AFDB, Test')

%%


[preds,scores] = predict(classification50Tree128ADASYN,trainMatrix128);

%[preds1,scores1] = predict(classificationRBFSVM128ADASYN,trainMatrix);

% Majority vote
[x,y,~,auc] = perfcurve(MM2,scores(:,2),1);
%[x1,y1,~,auc1] = perfcurve(MM,scores1(:,2),1);


figure()
plot(x,y,'LineWidth',2)
%hold on
%plot(x1,y1,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, AFDB, Test')

figure()
confusionchart(MM2,preds);
title('AFDB, Test')

%figure()
%confusionchart(MM,preds1);
%title('Cubic SVM, AFDB, Test')
