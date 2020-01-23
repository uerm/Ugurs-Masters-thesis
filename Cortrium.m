%% Load ECG signals
clear,clc
addpath 'C:\Users\Dell\Desktop\Test\mit-bih-arrhythmia-database-1.0.0' % Add path to WFDB toolbox

% MITDB Data
Data = [16121602, 16122101, 17011101, 17011102, 17011702, 17011703,...
    17012402, 17012404, 17012501, 17012502, 17013001, 17020702, 17020703,...
    17020801, 17021001, 17021002, 17021003, 17021004, 17021006, 17021007,...
    17022201, 17022204, 17030101, 17030102, 17030201, 17030202, 17030203,...
    17031001, 17031301, 17031402, 17070401, 17070402, 17071101, 17071102,...
    17071104];


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


%% Make a global annotation variable

% signal_length1 = length(sig1{2});
% signal_length2 = length(sig2{2});
% signal_length3 = length(sig3{2});
% anno_sig_1 = zeros(1,signal_length1);
% anno_sig_2 = zeros(1,signal_length2);
% anno_sig_3 = zeros(1,signal_length3);
% 
% ann1_ = [ann1{2}; [signal_length1]];
% ann2_ = [ann2{2}; [signal_length2]];
% ann3_ = [ann3{2}; [signal_length3]];
% 
% for i=1:length(comments1{2})
%     if strcmp(comments1{2}{i},'(AFIB')
%         anno_sig_1(ann1_(i):ann1_(i+1)) = 1;
%     end
% end
% for i=1:length(comments2{2})
%     if strcmp(comments2{2}{i},'(AFIB')
%         anno_sig_2(ann2_(i):ann2_(i+1)) = 1;
%     end
% end
% for i=1:length(comments3{2})
%     if strcmp(comments3{2}{i},'(AFIB')
%         anno_sig_3(ann3_(i):ann3_(i+1)) = 1;
%     end
% end
% 
% global_ann_sig = or(or(anno_sig_1,anno_sig_2),anno_sig_3);

%%
global_ann_sig = globalANN(sig1,sig2,sig3,ann1,ann2,ann3,...
    comments1,comments2,comments3,Data);