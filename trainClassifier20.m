function T = trainClassifier20(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM20_all = [];
predsLinSVM20_all = [];

targetsQuadSVM20_all = [];
predsQuadSVM20_all = [];

targetsCubicSVM20_all = [];
predsCubicSVM20_all = [];

targetsFineSVM20_all = [];
predsFineSVM20_all = [];

targetsMediumSVM20_all = [];
predsMediumSVM20_all = [];

targetsCoarseSVM20_all = [];
predsCoarseSVM20_all = [];

targets3KNN20_all = [];
preds3KNN20_all = [];

targets5KNN20_all = [];
preds5KNN20_all = [];

targets7KNN20_all = [];
preds7KNN20_all = [];

targets9KNN20_all = [];
preds9KNN20_all = [];

targets5Tree20_all = [];
preds5Tree20_all = [];

targets10Tree20_all = [];
preds10Tree20_all = [];

targets20Tree20_all = [];
preds20Tree20_all = [];

targets30Tree20_all = [];
preds30Tree20_all = [];

targets40Tree20_all = [];
preds40Tree20_all = [];

targets50Tree20_all = [];
preds50Tree20_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM20 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM20,~] = predict(classificationLinearSVM20,trainingData(test,1:end-1));
    targetLinSVM20 = trainingData(test,end);
    targetsLinSVM20_all = [targetsLinSVM20_all; squeeze(targetLinSVM20)];
    predsLinSVM20_all = [predsLinSVM20_all; squeeze(predsLinSVM20)];
    
    [~,scoresLinSVM20] = resubPredict(fitPosterior(classificationLinearSVM20));
    [xLinSVM20,yLinSVM20,~,aucLinSVM20] = perfcurve(trainingData(train,end),scoresLinSVM20(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM20.mat','classificationLinearSVM20','-v7.3'); % majority voting
    %save('classificationLinearSVM20_30.mat','classificationLinearSVM20','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM20_20.mat','classificationLinearSVM20','-v7.3'); % Threshold 20%
    save('classificationLinearSVM20_10.mat','classificationLinearSVM20','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM20 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM20,~] = predict(classificationQuadSVM20,trainingData(test,1:end-1));
    targetQuadSVM20 = trainingData(test,end);
    targetsQuadSVM20_all = [targetsQuadSVM20_all; squeeze(targetQuadSVM20)];
    predsQuadSVM20_all = [predsQuadSVM20_all; squeeze(predsQuadSVM20)];
    
    [~,scoresQuadSVM20] = resubPredict(fitPosterior(classificationQuadSVM20));
    [xQuadSVM20,yQuadSVM20,~,aucQuadSVM20] = perfcurve(trainingData(train,end),scoresQuadSVM20(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM20.mat','classificationQuadSVM20','-v7.3');
    %save('classificationQuadSVM20_30.mat','classificationQuadSVM20','-v7.3');
    %save('classificationQuadSVM20_20.mat','classificationQuadSVM20','-v7.3');
    save('classificationQuadSVM20_10.mat','classificationQuadSVM20','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM20 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM20,~] = predict(classificationCubicSVM20,trainingData(test,1:end-1));
    targetCubicSVM20 = trainingData(test,end);
    targetsCubicSVM20_all = [targetsCubicSVM20_all; squeeze(targetCubicSVM20)];
    predsCubicSVM20_all = [predsCubicSVM20_all; squeeze(predsCubicSVM20)];
    
    [~,scoresCubicSVM20] = resubPredict(fitPosterior(classificationCubicSVM20));
    [xCubicSVM20,yCubicSVM20,~,aucCubicSVM20] = perfcurve(trainingData(train,end),scoresCubicSVM20(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM20.mat','classificationCubicSVM20','-v7.3');
    %save('classificationCubicSVM20_30.mat','classificationCubicSVM20','-v7.3');
    %save('classificationCubicSVM20_20.mat','classificationCubicSVM20','-v7.3');
    save('classificationCubicSVM20_10.mat','classificationCubicSVM20','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM20 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM20,~] = predict(classificationFineSVM20,trainingData(test,1:end-1));
    targetsFineSVM20 = trainingData(test,end);
    targetsFineSVM20_all = [targetsFineSVM20_all; squeeze(targetsFineSVM20)];
    predsFineSVM20_all = [predsFineSVM20_all; squeeze(predsFineSVM20)];
    
    [~,scoresFineSVM20] = resubPredict(fitPosterior(classificationFineSVM20));
    [xFineSVM20,yFineSVM20,~,aucFineSVM20] = perfcurve(trainingData(train,end),scoresFineSVM20(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM20.mat','classificationFineSVM20','-v7.3');
    %save('classificationFineSVM20_30.mat','classificationFineSVM20','-v7.3');
    %save('classificationFineSVM20_20.mat','classificationFineSVM20','-v7.3');
    save('classificationFineSVM20_10.mat','classificationFineSVM20','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM20 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM20,~] = predict(classificationMediumSVM20,trainingData(test,1:end-1));
    targetsMediumSVM20 = trainingData(test,end);
    targetsMediumSVM20_all = [targetsMediumSVM20_all; squeeze(targetsMediumSVM20)];
    predsMediumSVM20_all = [predsMediumSVM20_all; squeeze(predsMediumSVM20)];
    
    [~,scoresMediumSVM20] = resubPredict(fitPosterior(classificationMediumSVM20));
    [xMediumSVM20,yMediumSVM20,~,aucMediumSVM20] = perfcurve(trainingData(train,end),scoresMediumSVM20(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM20.mat','classificationMediumSVM20','-v7.3');
    %save('classificationMediumSVM20_30.mat','classificationMediumSVM20','-v7.3');
    %save('classificationMediumSVM20_20.mat','classificationMediumSVM20','-v7.3');
    save('classificationMediumSVM20_10.mat','classificationMediumSVM20','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM20 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM20,~] = predict(classificationCoarseSVM20,trainingData(test,1:end-1));
    targetsCoarseSVM20 = trainingData(test,end);
    targetsCoarseSVM20_all = [targetsCoarseSVM20_all; squeeze(targetsCoarseSVM20)];
    predsCoarseSVM20_all = [predsCoarseSVM20_all; squeeze(predsCoarseSVM20)];
    
    [~,scoresCoarseSVM20] = resubPredict(fitPosterior(classificationCoarseSVM20));
    [xCoarseSVM20,yCoarseSVM20,~,aucCoarseSVM20] = perfcurve(trainingData(train,end),scoresCoarseSVM20(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM20.mat','classificationCoarseSVM20','-v7.3');
    %save('classificationCoarseSVM20_30.mat','classificationCoarseSVM20','-v7.3');
    %save('classificationCoarseSVM20_20.mat','classificationCoarseSVM20','-v7.3');
    save('classificationCoarseSVM20_10.mat','classificationCoarseSVM20','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN20 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN20,~] = predict(classification3KNN20,trainingData(test,1:end-1));
    targets3KNN20 = trainingData(test,end);
    targets3KNN20_all = [targets3KNN20_all; squeeze(targets3KNN20)];
    preds3KNN20_all = [preds3KNN20_all; squeeze(preds3KNN20)];
    
    [~,scores3KNN20] = resubPredict((classification3KNN20));
    [x3KNN20,y3KNN20,~,auc3KNN20] = perfcurve(trainingData(train,end),scores3KNN20(:,2),1);
    t7 = toc;
    
    %save('classification3KNN20.mat','classification3KNN20','-v7.3');
    %save('classification3KNN20_30.mat','classification3KNN20','-v7.3');
    %save('classification3KNN20_20.mat','classification3KNN20','-v7.3');
    save('classification3KNN20_10.mat','classification3KNN20','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN20 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN20,~] = predict(classification5KNN20,trainingData(test,1:end-1));
    targets5KNN20 = trainingData(test,end);
    targets5KNN20_all = [targets5KNN20_all; squeeze(targets5KNN20)];
    preds5KNN20_all = [preds5KNN20_all; squeeze(preds5KNN20)];
    
    [~,scores5KNN20] = resubPredict((classification5KNN20));
    [x5KNN20,y5KNN20,~,auc5KNN20] = perfcurve(trainingData(train,end),scores5KNN20(:,2),1);
    t8 = toc;

    %save('classification5KNN20.mat','classification5KNN20','-v7.3');
    %save('classification5KNN20_30.mat','classification5KNN20','-v7.3');
    %save('classification5KNN20_20.mat','classification5KNN20','-v7.3');
    save('classification5KNN20_10.mat','classification5KNN20','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN20 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN20,~] = predict(classification7KNN20,trainingData(test,1:end-1));
    targets7KNN20 = trainingData(test,end);
    targets7KNN20_all = [targets7KNN20_all; squeeze(targets7KNN20)];
    preds7KNN20_all = [preds7KNN20_all; squeeze(preds7KNN20)];
    
    [~,scores7KNN20] = resubPredict((classification7KNN20));
    [x7KNN20,y7KNN20,~,auc7KNN20] = perfcurve(trainingData(train,end),scores7KNN20(:,2),1);
    t9 = toc;
    
    %save('classification7KNN20.mat','classification7KNN20','-v7.3');
    %save('classification7KNN20_30.mat','classification7KNN20','-v7.3');
    %save('classification7KNN20_20.mat','classification7KNN20','-v7.3');
    save('classification7KNN20_10.mat','classification7KNN20','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN20 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN20,~] = predict(classification9KNN20,trainingData(test,1:end-1));
    targets9KNN20 = trainingData(test,end);
    targets9KNN20_all = [targets9KNN20_all; squeeze(targets9KNN20)];
    preds9KNN20_all = [preds9KNN20_all; squeeze(preds9KNN20)];
    
    [~,scores9KNN20] = resubPredict((classification9KNN20));
    [x9KNN20,y9KNN20,~,auc9KNN20] = perfcurve(trainingData(train,end),scores9KNN20(:,2),1);
    t10 = toc;
    
    %save('classification9KNN20.mat','classification9KNN20','-v7.3');
    %save('classification9KNN20_30.mat','classification9KNN20','-v7.3');
    %save('classification9KNN20_20.mat','classification9KNN20','-v7.3');
    save('classification9KNN20_10.mat','classification9KNN20','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree20 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree20,~] = predict(classification5Tree20,trainingData(test,1:end-1));
    targets5Tree20 = trainingData(test,end);
    targets5Tree20_all = [targets5Tree20_all; squeeze(targets5Tree20)];
    preds5Tree20_all = [preds5Tree20_all; squeeze(preds5Tree20)];
    
    [~,scores5Tree20] = resubPredict((classification5Tree20));
    [x5Tree20,y5Tree20,~,auc5Tree20] = perfcurve(trainingData(train,end),scores5Tree20(:,2),1);
    t11 = toc;
    
    %save('classification5Tree20.mat','classification5Tree20','-v7.3');
    %save('classification5Tree20_30.mat','classification5Tree20','-v7.3');
    %save('classification5Tree20_20.mat','classification5Tree20','-v7.3');
    save('classification5Tree20_10.mat','classification5Tree20','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree20 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree20,~] = predict(classification10Tree20,trainingData(test,1:end-1));
    targets10Tree20 = trainingData(test,end);
    targets10Tree20_all = [targets10Tree20_all; squeeze(targets10Tree20)];
    preds10Tree20_all = [preds10Tree20_all; squeeze(preds10Tree20)];
    
    [~,scores10Tree20] = resubPredict((classification10Tree20));
    [x10Tree20,y10Tree20,~,auc10Tree20] = perfcurve(trainingData(train,end),scores10Tree20(:,2),1);
    t12 = toc;
    
    %save('classification10Tree20.mat','classification10Tree20','-v7.3');
    %save('classification10Tree20_30.mat','classification10Tree20','-v7.3');
    %save('classification10Tree20_20.mat','classification10Tree20','-v7.3');
    save('classification10Tree20_10.mat','classification10Tree20','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree20 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree20,~] = predict(classification20Tree20,trainingData(test,1:end-1));
    targets20Tree20 = trainingData(test,end);
    targets20Tree20_all = [targets20Tree20_all; squeeze(targets20Tree20)];
    preds20Tree20_all = [preds20Tree20_all; squeeze(preds20Tree20)];
    [~,scores20Tree20] = resubPredict((classification20Tree20));
    
    [x20Tree20,y20Tree20,~,auc20Tree20] = perfcurve(trainingData(train,end),scores20Tree20(:,2),1);
    t13 = toc;
    
    %save('classification20Tree20.mat','classification20Tree20','-v7.3');
    %save('classification20Tree20_30.mat','classification20Tree20','-v7.3');
    %save('classification20Tree20_20.mat','classification20Tree20','-v7.3');
    save('classification20Tree20_10.mat','classification20Tree20','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree20 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree20,~] = predict(classification30Tree20,trainingData(test,1:end-1));
    targets30Tree20 = trainingData(test,end);
    targets30Tree20_all = [targets30Tree20_all; squeeze(targets30Tree20)];
    preds30Tree20_all = [preds30Tree20_all; squeeze(preds30Tree20)];
    
    [~,scores30Tree20] = resubPredict((classification30Tree20));
    [x30Tree20,y30Tree20,~,auc30Tree20] = perfcurve(trainingData(train,end),scores30Tree20(:,2),1);
    t14 = toc;
    
    %save('classification30Tree20.mat','classification30Tree20','-v7.3');
    %save('classification30Tree20_30.mat','classification30Tree20','-v7.3');
    %save('classification30Tree20_20.mat','classification30Tree20','-v7.3');
    save('classification30Tree20_10.mat','classification30Tree20','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree20 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree20,~] = predict(classification40Tree20,trainingData(test,1:end-1));
    targets40Tree20 = trainingData(test,end);
    targets40Tree20_all = [targets40Tree20_all; squeeze(targets40Tree20)];
    preds40Tree20_all = [preds40Tree20_all; squeeze(preds40Tree20)];
    
    [~,scores40Tree20] = resubPredict((classification40Tree20));
    [x40Tree20,y40Tree20,~,auc40Tree20] = perfcurve(trainingData(train,end),scores40Tree20(:,2),1);
    t15 = toc;
    
    %save('classification40Tree20.mat','classification40Tree20','-v7.3');
    %save('classification40Tree20_30.mat','classification40Tree20','-v7.3');
    %save('classification40Tree20_20.mat','classification40Tree20','-v7.3');
    save('classification40Tree20_10.mat','classification40Tree20','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree20 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree20,~] = predict(classification50Tree20,trainingData(test,1:end-1));
    targets50Tree20 = trainingData(test,end);
    targets50Tree20_all = [targets50Tree20_all; squeeze(targets50Tree20)];
    preds50Tree20_all = [preds50Tree20_all; squeeze(preds50Tree20)];
    
    [~,scores50Tree20] = resubPredict((classification50Tree20));
    [x50Tree20,y50Tree20,~,auc50Tree20] = perfcurve(trainingData(train,end),scores50Tree20(:,2),1);
    t16 = toc;
    
    %save('classification50Tree20.mat','classification50Tree20','-v7.3');
    %save('classification50Tree20_30.mat','classification50Tree20','-v7.3');
    %save('classification50Tree20_20.mat','classification50Tree20','-v7.3');
    save('classification50Tree20_10.mat','classification50Tree20','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM20_all,predsLinSVM20_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM20_all,predsQuadSVM20_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM20_all,predsCubicSVM20_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM20_all,predsFineSVM20_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM20_all,predsMediumSVM20_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM20_all,predsCoarseSVM20_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN20_all,preds3KNN20_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN20_all,preds5KNN20_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN20_all,preds7KNN20_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN20_all,preds9KNN20_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree20_all,preds5Tree20_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree20_all,preds10Tree20_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree20_all,preds20Tree20_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree20_all,preds30Tree20_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree20_all,preds40Tree20_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree20_all,preds50Tree20_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM20,yLinSVM20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM20,yQuadSVM20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM20,yCubicSVM20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM20,yFineSVM20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM20,yMediumSVM20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM20,yCoarseSVM20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN20,y3KNN20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN20,y5KNN20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN20,y7KNN20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN20,y9KNN20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree20,y5Tree20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree20,y10Tree20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree20,y20Tree20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree20,y30Tree20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree20,y40Tree20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree20,y50Tree20,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 50 Trees')
t48 = toc;



% Training time
Time = [t1;t2;t3;t4;t5;t6;t7;t8;t9;t10;t11;t12;t13;t14;t15;t16];
Time_total = [t1+t17+t33; t2+t18+t34; t3+t19+t35; t4+t20+t36; t5+t21+t37;...
    t6+t22+t38; t7+t23+t39; t8+t24+t40; t9+t25+t41; t10+t26+t42;...
    t11+t27+t43; t12+t28+t44; t13+t29+t45; t14+t30+t46; t15+t31+t47;...
    t16+t32+t48];
AUC = [aucLinSVM20;aucQuadSVM20;aucCubicSVM20;aucFineSVM20;aucMediumSVM20;...
    aucCoarseSVM20;auc3KNN20;auc5KNN20;auc7KNN20;auc9KNN20;auc5Tree20;...
    auc10Tree20;auc20Tree20;auc30Tree20;auc40Tree20;auc50Tree20];

T = table(AUC,Time,Time_total);

