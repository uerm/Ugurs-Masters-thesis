function [m_RRIseg, s_RRIseg, r_RRIseg, n_RRIseg, CV_RRIseg, minRRIseg,...
   m_RRIseg1, s_RRIseg1, r_RRIseg1, n_RRIseg1, CV_RRIseg1, minRRIseg1,...
   trainMatrix] = FeatureExtraction(qrs,seg,data,Fs)

D = length(data);

RRI = zeros(0);

% RR-interval without segmentation.
for i = 1:D
    RRI{1,i} = diff(qrs{1,i})/Fs;
end

% RR-interval with segmentation[
%fun = @(m)diff(m,1,2);
RRIseg = cellfun(@(m)diff(m,1,2),seg,'uni',0);
for i = 1:length(RRIseg)
    RRIseg{i} = RRIseg{i}/Fs;
end

M_RRI = zeros(0);
SDNN = zeros(0);
RMSSD = zeros(0);
nRMSSD = zeros(0);
NN50 = zeros(0);
pNN50 = zeros(0);
CV = zeros(0);
minRRI = zeros(0);

for i = 1:D
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
        if (RRI{1,i}(num+1)-RRI{1,i}(num) > 50*10^(-3)*180)
            n = n+1;
        end
    end
    NN50{i} = n;
end

% pNN50 (in percentage)
for i = 1:length(RRI)
    pNN50{1,i} = (NN50{1,i}/length(RRI{1,i}))*100;
end

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
