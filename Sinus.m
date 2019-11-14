x = 0:0.02:7*pi;
a = sin(x);
%plot(x,a);
sineTrain = repmat(a,100,1);
sineTest = repmat(a,100,1);

% for i = 1:10000
%     sine{1,i} = sin(x);
% end


%XTrain = squeeze(resh{1}(11,:,:));
%XTest = squeeze(resh{1}(12,:,:));