%% Load ECG signals
clear,clc
addpath 'C:\Users\Dell\Desktop\Test\mit-bih-arrhythmia-database-1.0.0' % Add path to WFDB toolbox

% MITDB Data
Data = [103, 104, 105, 118, 123, 202, 205, 210, 212, 219];

for i = 1:length(Data)
    [sig1{i}, Fs1, ~] = rdsamp(num2str(Data(i)),1); % Lead I
    [sig2{i}, Fs2, ~] = rdsamp(num2str(Data(i)),2); % Lead II
end


%% Denoising with DWT - Lead I and Lead II
tic
y1 = dwt_denoise1(sig1,8,Data); % Denoising Lead I
y2 = dwt_denoise2(sig2,8,Data); % Denoising Lead II
toc

%% Downsampling
[p,q] = rat(125/360);

for i = 1:length(Data)
    y1_new{i} = resample(y1{i}, p, q);
    y2_new{i} = resample(y2{i}, p, q);
end
clear y1 y2 p q

%% Read annotations

for i = 1:length(Data)
    [ann2{i},anntype2{i},~,~,~,comments{i}] = rdann(num2str(Data(i)),'atr');
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
    QRS2{1,i} = round(QRS2{1,i}.*(125/360));
    ann2{1,i} = round(ann2{1,i}.*(125/360));
    wave2{1,i} = round(wave2{1,i}.*(125/360));
end
toc
%%
tic

% "Cut" the labels according to the R peaks
anntype_new = annotation2(Data,loc2,wave2,ann2,comments2);

AL = labelling2(anntype_new,Data); % Label the data

toc


%%
% % For each cell of AL, find the non-2's
% keepIndices = cellfun(@(x)x~=2,AL,'UniformOutput',false);
% 
% % Keep the elements of AL that are non-2's
% AL2  = cellfun(@(x,y)x(y),AL, keepIndices,'UniformOutput',false);
% 
% % Keep the elements of QRS that are non-2's
% QRS2 = cellfun(@(x,y)x(y),QRS,keepIndices,'UniformOutput',false);



%%
tic

AL_128 = segmentation(AL,20); % Define segment length

%M = threshold(AL_128,30); % Threshold 30%

% Use the "M" variables depending on the threshold that will be used.

M = cellfun(@(m)mode(m,2), AL_128,'uni',0); % Majority voting

% Stack the labels for classification
MM = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M, 2), 'UniformOutput', false));

toc


%% Plot of subject 48 - both leads
subplot(211)
plot(y1_new{1,48})
title('Subject 234, lead I')
xlim([0 4000])
xlabel('Samples')
ylabel('Amplitude (mV)')

subplot(212)
plot(y2_new{1,48})
title('Subject 234, lead II')
xlim([0 4000])
xlabel('Samples')
ylabel('Amplitude (mV)')


%% Result of DWT filtering - plots
subplot(221)
plot(sig1{1,48})
title('Subject 234, Lead I')
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')


subplot(222)
plot(y1{1,48})
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 234, Lead I')
legend('DWT Filtering','Location','Best')

subplot(223)
plot(sig2{1,48})
title('Subject 234, Lead II')
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
legend('No Filtering','Location','Best')

subplot(224)
plot(y2{1,48})
xlim([0 7200])
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 234, Lead II')
legend('DWT Filtering','Location','Best')

%%
% 
% y1_new = zeros(0);
% y2_new = zeros(0);
% 
% for i = 1:length(Data)
%     y1_new{i} = y1{i};
%     y2_new{i} = y2{i};
% 
%     y1_new{i}(y1_new{i}> 1.5) = 1.5;
%     y2_new{i}(y2_new{i}> 1.5) = 1.5;
% end

%% Segment the R peaks in 128 R peaks per segment
% Use the segmentation helper function
% segmentation(data,n), where data is the index of the R peaks and n is the
% length of the segment.
Data_10 = segmentation(QRS2,20);

%Data_10_2 = Data_10(~cellfun('isempty',Data_10));



%% RR-interval and HRV features
% Use the FeatureExtraction helper function to extract HRV features.

Fs = Fs1*(125/Fs1);

[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~,...
   trainMatrix] = FeatureExtraction(QRS2,Data_10,Data,Fs);


%% Segmentation of filtered signal based on R peak location
subplot(211)
plot(y1_new{1,1}(1:Data_10{1}(1,end)))
title('Segmentation (first 10 R peaks), subject 1, lead I, no zero padding')
xlabel('Samples')
ylabel('Amplitude (mV)')
subplot(212)
plot(y2_new{1,1}(1:Data_10{1}(1,end)))
title('Segmentation (first 10 R peaks), subject 1, lead II, no zero padding')
xlabel('Samples')
ylabel('Amplitude (mV)')

%% Cutting the filtered signal in segments - Both leads, all patients
% Constructing tensor for each patient


[ecg_segments1, ecg_segments2, ~] = TensorConstruct(y1_new,y2_new,Data_10);
 
ecg1 = cellfun(@(m) normalize(m,2), ecg_segments1,'uni',0);
ecg2 = cellfun(@(m) normalize(m,2), ecg_segments2,'uni',0);


% for i = 1:length(Data)
%     ecg1{i} = (ecg_segments1{i} - mean(ecg_segments1{i},2));
%     ecg2{i} = (ecg_segments2{i} - mean(ecg_segments2{i},2));
% end

for i = 1:length(Data)
    tensor{i} = cat(3,ecg1{i},ecg2{i});
end

%% Plot of ECG with zero padding
figure()
subplot(211)
plot(ecg_segments1{1,1}(1,:))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Segmentation (first 20 R peaks), subject 100, Lead I, with zero padding')
subplot(212)
plot(ecg_segments2{1,1}(1,:))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Segmentation (first 20 R peaks), subject 100, Lead II, with zero padding')

figure()
subplot(211)
plot(ecg_segments1{1,1}(1,1800:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 100, Lead I, with zeros')
subplot(212)
plot(ecg_segments2{1,1}(1,1800:end))
xlabel('Samples')
ylabel('Amplitude (mV)')
title('Subject 100, Lead II, with zeros')


%% Wavelet + EMD
tic
% clear y1 y2 sig1_new sig2_new ...
%     twaves pwaves
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
toc
%% Reshape EMDs
tic 
% Reshape all the perm matrices
for i = 1:length(perm)
    resh{1,i} = reshape(perm{1,i},[],size(perm{1,i},5),size(perm{1,i},4));
end
toc

%%

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

parfor i = 1:size(skew,1)
    i
    skew(i,:,:) = double(skewness(stacked(i,:,:),1,2));
    kurt(i,:,:) = double(kurtosis(stacked(i,:,:),1,2));
end
toc

skew = squeeze(skew);
kurt = squeeze(kurt);

%%
% All statistical features concatenated into one variable.
feat_cat = cat(1,mu,st,v,skew,kurt);

%% Train with only HRV features (also with ADASYN)

% No ADASYN
trainMatrix1 = [trainMatrix MM];

feat1 = trainMatrix1(:,1:end-1);

MM1 = trainMatrix1(:,end);

[out_featuresSyn1, out_labelsSyn1] = ADASYN(feat1, MM1, 1,...
    5, 5, true);

t1 = [out_featuresSyn1 double(out_labelsSyn1)];

% ADASYN
trainMatrix11 = [trainMatrix1;t1];


%% Train with only DWT+EMD statistical features (also with ADASYN)

% Statistical features transposed to match the other features
feat_cat20 = feat_cat';

% No ADASYN
trainMatrix2 = [feat_cat MM];

feat2 = trainMatrix2(:,1:end-1);

MM2 = trainMatrix2(:,end);

[out_featuresSyn2, out_labelsSyn2] = ADASYN(feat2, MM2, 1,...
    5, 5, true);

t2 = [out_featuresSyn2 double(out_labelsSyn2)];

% ADASYN
trainMatrix22 = [trainMatrix2;t2];

%% Train with HRV features + DWT+EMD features (also with ADASYN)

% No ADASYN
trainMatrix3 = [feat_cat20 trainMatrix MM];

feat3 = trainMatrix3(:,1:end-1);

MM3 = trainMatrix3(:,end);

[out_featuresSyn3, out_labelsSyn3] = ADASYN(feat3, MM3, 1,...
    5, 5, true);

t3 = [out_featuresSyn3 double(out_labelsSyn3)];

% ADASYN
trainMatrix33 = [trainMatrix3;t3];

%% Create scatter plots of data

x = trainMatrix3(:,452);
y = trainMatrix3(:,451);
scatterhist(x,y,'Group',trainMatrix3(:,end),'Kernel','on','Location','SouthEast',...
    'Direction','out','Color','br')
xlabel('Standard Deviation of RRi (F452)')
ylabel('Mean RRi (F451)')
title('MITDB, majority vote, M = 128 beats, after ADASYN')

%%
x = trainMatrix33(:,452);
y = trainMatrix33(:,451);
scatterhist(x,y,'Group',trainMatrix33(:,end),'Kernel','on','Location','SouthEast',...
    'Direction','out','Color','br')
xlabel('Standard Deviation of RRi (F452)')
ylabel('Mean RRi (F451)')
title('MITDB, majority vote, M = 20 beats, after ADASYN')


%% Testing - confusion matrix and predictions


trainMatrix4 = trainMatrix3;


% Linear SVM
[preds1,scores1] = predict(classificationLinearSVM20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds1);
title('MITDB, Linear SVM, Test')

% Quadratic SVM
[preds2,scores2] = predict(classificationQuadSVM20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds2);
title('MITDB, Quadratic SVM, Test')

% Cubic SVM
[preds3,scores3] = predict(classificationCubicSVM20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds3);
title('MITDB, Cubic SVM, Test')


% RBF SVM
[preds4,scores4] = predict(classificationRBFSVM20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds4);
title('MITDB, RBF SVM, Test')


% 3 kNN
[preds5,scores5] = predict(classification3KNN20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds5);
title('MITDB, kNN - k = 3, Test')


% 5 kNN
[preds6,scores6] = predict(classification5KNN20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds6);
title('MITDB, kNN - k = 5, Test')


% 7 kNN
[preds7,scores7] = predict(classification7KNN20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds7);
title('MITDB, kNN - k = 7, Test')


% 9 kNN
[preds8,scores8] = predict(classification9KNN20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds8);
title('MITDB, kNN - k = 9, Test')


% RF, 5 Trees
[preds9,scores9] = predict(classification5Tree20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds9);
title('MITDB, RF - 5 Trees, Test')


% RF, 10 Trees
[preds10,scores10] = predict(classification10Tree20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds10);
title('MITDB, RF - 10 Trees, Test')


% RF, 20 Trees
[preds11,scores11] = predict(classification20Tree20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds11);
title('MITDB, RF - 20 Trees, Test')


% RF, 30 Trees
[preds12,scores12] = predict(classification30Tree20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds12);
title('MITDB, RF - 30 Trees, Test')


% RF, 40 Trees
[preds13,scores13] = predict(classification40Tree20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds13);
title('MITDB, RF - 40 Trees, Test')


% RF, 50 Trees
[preds14,scores14] = predict(classification50Tree20ADASYN,trainMatrix4(:,1:end-1));

figure()
confusionchart(trainMatrix4(:,end),preds14);
title('MITDB, RF - 50 Trees, Test')





%% ROC plots

[x1,y1,~,auc1] = perfcurve(trainMatrix4(:,end),scores1(:,2),1);

figure()
plot(x1,y1,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, Linear SVM, Test')


[x2,y2,~,auc2] = perfcurve(trainMatrix4(:,end),scores2(:,2),1);

figure()
plot(x2,y2,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, Quadratic SVM, Test')


[x3,y3,~,auc3] = perfcurve(trainMatrix4(:,end),scores3(:,2),1);

figure()
plot(x3,y3,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, Cubic SVM, Test')


[x4,y4,~,auc4] = perfcurve(trainMatrix4(:,end),scores4(:,2),1);

figure()
plot(x4,y4,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RBF SVM, Test')


[x5,y5,~,auc5] = perfcurve(trainMatrix4(:,end),scores5(:,2),1);

figure()
plot(x5,y5,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, kNN - k = 3, Test')


[x6,y6,~,auc6] = perfcurve(trainMatrix4(:,end),scores6(:,2),1);

figure()
plot(x6,y6,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, kNN - k = 5, Test')


[x7,y7,~,auc7] = perfcurve(trainMatrix4(:,end),scores7(:,2),1);

figure()
plot(x7,y7,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, kNN - k = 7, Test')



[x8,y8,~,auc8] = perfcurve(trainMatrix4(:,end),scores8(:,2),1);

figure()
plot(x8,y8,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, kNN - k = 9, Test')



[x9,y9,~,auc9] = perfcurve(trainMatrix4(:,end),scores9(:,2),1);

figure()
plot(x9,y9,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RF 5 Trees, Test')


[x10,y10,~,auc10] = perfcurve(trainMatrix4(:,end),scores10(:,2),1);

figure()
plot(x10,y10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RF 10 Trees, Test')



[x11,y11,~,auc11] = perfcurve(trainMatrix4(:,end),scores11(:,2),1);

figure()
plot(x11,y11,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RF 20 Trees, Test')


[x12,y12,~,auc12] = perfcurve(trainMatrix4(:,end),scores12(:,2),1);

figure()
plot(x12,y12,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RF 30 Trees, Test')



[x13,y13,~,auc13] = perfcurve(trainMatrix4(:,end),scores13(:,2),1);

figure()
plot(x13,y13,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RF 40 Trees, Test')


[x14,y14,~,auc14] = perfcurve(trainMatrix4(:,end),scores14(:,2),1);

figure()
plot(x14,y14,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, MITDB, RF 50 Trees, Test')


AUC = [auc1;auc2;auc3;auc4;auc5;auc6;auc7;auc8;auc9;auc10;auc11;auc12;...
    auc13;auc14];

T_AUC = table(AUC);








