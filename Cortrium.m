%% Load ECG signals
clear,clc
addpath 'C:\Users\Dell\Desktop\Test\mit-bih-arrhythmia-database-1.0.0' % Add path to WFDB toolbox

% MITDB Data
Data = [16121602, 16122101, 17011101, 17011102, 17011703,...
    17012404, 17012501, 17012502, 17013001, 17020702, 17020703,...
    17020801, 17021001, 17021002, 17021003, 17021004, 17021006,...
    17022201, 17022204, 17030101, 17030102, 17030201, 17030202,...
    17031001, 17031301, 17031402, 17070401, 17070402, 17071101, 17071102,...
    17071104];

% Discarded records
%17011702, (5)
%17012402, (7)
%17021007, (20)
%17030203, (27)


for i = 1:length(Data)
    [sig1{i}, Fs1, tm1{i}] = rdsamp(num2str(Data(i)),1); % Lead I
    [sig2{i}, Fs2, tm2{i}] = rdsamp(num2str(Data(i)),2); % Lead II
    [sig3{i}, Fs3, tm3{i}] = rdsamp(num2str(Data(i)),3); % Lead III
end

%% Read annotations

for i = 1:length(Data)
    [ann1{i},anntype1{i},subtype1{i},chan1{i},num1{i},comments1{i}] = rdann(num2str(Data(i)),'atr',1); % Lead I
    [ann2{i},anntype2{i},subtype2{i},chan2{i},num2{i},comments2{i}] = rdann(num2str(Data(i)),'atr',2); % Lead II
    [ann3{i},anntype3{i},subtype3{i},chan3{i},num3{i},comments3{i}] = rdann(num2str(Data(i)),'atr',3); % Lead III
end

%% Find R peaks and segment R peaks with 50 % overlap


for i = 1:length(Data)
    ecgpuwave(num2str(Data(i)),'test');
    pwaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'p');
    twaves{i} = rdann(num2str(Data(i)),'test',[],[],[],'t');
    QRS{i} = rdann(num2str(Data(i)),'test',[],[],[],'N');
    [wave{i},loc{i}] = rdann(num2str(Data(i)),'test',[],[],[],'');
end

%%
QRS = {QRS1{1,1},QRS2{1,2},QRS3{1,3},QRS4{1,4},QRS6{1,6},QRS8{1,8},QRS9{1,9},...
    QRS10{1,10},QRS11{1,11}, QRS12{1,12},QRS13{1,13},QRS14{1,14},QRS15{1,15},...
    QRS16{1,16},QRS17{1,17},QRS18{1,18},QRS19{1,19},QRS21{1,21},...
    QRS22{1,22},QRS23{1,23},QRS24{1,24},QRS25{1,25},QRS26{1,26},...
    QRS28{1,28},QRS29{1,29},QRS30{1,30},QRS31{1,31},QRS32{1,32},QRS33{1,33},...
    QRS34{1,34},QRS35{1,35}};

loc = {loc1{1,1},loc2{1,2},loc3{1,3},loc4{1,4},loc6{1,6},loc8{1,8},loc9{1,9},...
    loc10{1,10},loc11{1,11}, loc12{1,12},loc13{1,13},loc14{1,14},loc15{1,15},...
    loc16{1,16},loc17{1,17},loc18{1,18},loc19{1,19},loc21{1,21},...
    loc22{1,22},loc23{1,23},loc24{1,24},loc25{1,25},loc26{1,26},...
    loc28{1,28},loc29{1,29},loc30{1,30},loc31{1,31},loc32{1,32},loc33{1,33},...
    loc34{1,34},loc35{1,35}};

wave = {wave1{1,1},wave2{1,2},wave3{1,3},wave4{1,4},wave6{1,6},wave8{1,8},wave9{1,9},...
    wave10{1,10},wave11{1,11}, wave12{1,12},wave13{1,13},wave14{1,14},wave15{1,15},...
    wave16{1,16},wave17{1,17},wave18{1,18},wave19{1,19},wave21{1,21},...
    wave22{1,22},wave23{1,23},wave24{1,24},wave25{1,25},wave26{1,26},...
    wave28{1,28},wave29{1,29},wave30{1,30},wave31{1,31},wave32{1,32},wave33{1,33},...
    wave34{1,34},wave35{1,35}};


%% Make global annotation
global_ann_sig = globalANN(sig1,sig2,sig3,ann1,ann2,ann3,...
    comments1,comments2,comments3,Data);

%% Cut the signal to match the annotation

for i = 1:length(Data)
    ann{i} = sort(cat(1,ann1{i},ann2{i},ann3{i}));
end


for i = 1:length(Data)
    sig1_new{i} = sig1{i}(min(ann{i}):max(ann{i}));
    sig2_new{i} = sig2{i}(min(ann{i}):max(ann{i}));
    sig3_new{i} = sig3{i}(min(ann{i}):max(ann{i}));
    global_ann_sig2{i} = global_ann_sig{i}(min(ann{i}):max(ann{i}));
end

%% Cut the R peaks according to annotation
QRS2 = QRS;

for i = 1:length(Data)
    for j = 1:length(QRS2{i})
    if (QRS2{i}(j) < min(ann{i})) || (QRS2{i}(j) > max(ann{i}))
        QRS2{i}(j) = 0;
    end
    end
    QRS2{i} = nonzeros(QRS2{i});
    QRS3{i} = QRS2{i}-(min(QRS2{i})-1);
end

%% Cut annotation according to R peak indeces
for i = 1:length(Data)
    AL{i} = global_ann_sig2{i}(QRS3{i});
end

AL_128 = segmentation(AL,128); % Define segment length

%M = threshold(AL_128,10); % Threshold 30%

% Use the "M" variables depending on the threshold that will be used.

M = cellfun(@(m)mode(m,2), AL_128,'uni',0); % Majority voting

% Stack the labels for classification
MM = cell2mat(cellfun(@(col) vertcat(col{:}), num2cell(M, 2), 'UniformOutput', false));


%% Denoise signals

y1 = dwt_denoise1(sig1_new,8,Data); % Denoising Lead I
y2 = dwt_denoise2(sig2_new,8,Data); % Denoising Lead II
%y3 = dwt_denoise3(sig3_new,8,Data); % Denoising Lead III
%%

y1_new = zeros(0);
y2_new = zeros(0);

for i = 1:length(Data)
    y1_new{i} = y1{i};
    y2_new{i} = y2{i};

    y1_new{i}(y1_new{i}> 1.5) = 1.5;
    y2_new{i}(y2_new{i}> 1.5) = 1.5;
end


%% Segment R peaks
Data_10 = segmentation(QRS3,20);

%% Extract HRV features
[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, trainMatrix] =...
    FeatureExtraction(QRS3,Data_10,Data);

%% Construct tensor for all patients and with two leads
[~, ~, tensor] = TensorConstruct(y1_new,y2_new,Data_10);

%% Run EMD algorithm
tic
%clear ecg_segments1 ecg_segments2 ecg_segments3 Data_10 y1 y2 sig1_new sig2_new ...
%    twaves pwaves
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
    perm1{1,i} = permute(EMD{1,i},[2,4,5,1,3]);
end
toc
%% Reshape EMDs
tic 
% Reshape all the perm matrices
for i = 1:length(perm1)
    resh{1,i} = reshape(perm1{1,i},[],size(perm1{1,i},5),size(perm1{1,i},4));
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
stacked = cat(3,resh{:});
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


%% 

feat_cat = feat_cat';
trainMatrixC = [feat_cat trainMatrix];

%%

[preds,scores] = predict(classificationRBFSVM128ADASYN,trainMatrixC);




% Majority vote
[x,y,~,auc] = perfcurve(MM,scores(:,2),1);


figure()
plot(x,y,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM, AFDB, Test')

figure()
confusionchart(MM,preds);
title('Cubic SVM, AFDB, Test')



