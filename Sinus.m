

sineTrain = zeros(1100,200);
for i=1:200
    x = linspace(1,(rand(1)+1)*3*pi,1100);
    a = sin(x);
    sineTrain(:,i) = a;
end
sineTest = sineTrain(:,1:100);
sineTrain = sineTrain(:,101:200);


%XTrain = squeeze(resh{1}(11,:,:));
%XTest = squeeze(resh{1}(12,:,:));

sineTest = reshape(single(sineTest), 1, size(sineTest,1), 1, size(sineTest,2));
sineTrain = reshape(single(sineTrain), 1, size(sineTrain,1), 1, size(sineTrain,2));