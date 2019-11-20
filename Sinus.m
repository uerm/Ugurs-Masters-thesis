x = 0:0.02:7*pi;
a = sin(x);

sineTrain = repmat(a,100,1)';
sineTest = repmat(a,100,1)';


%XTrain = squeeze(resh{1}(11,:,:));
%XTest = squeeze(resh{1}(12,:,:));

sineTest = reshape(single(sineTest), 1, size(sineTest,1), 1, size(sineTest,2));
sineTrain = reshape(single(sineTrain), 1, size(sineTrain,1), 1, size(sineTrain,2));