function T = trainClassifier128(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM128_all = [];
predsLinSVM128_all = [];

targetsQuadSVM128_all = [];
predsQuadSVM128_all = [];

targetsCubicSVM128_all = [];
predsCubicSVM128_all = [];

targetsFineSVM128_all = [];
predsFineSVM128_all = [];

targetsMediumSVM128_all = [];
predsMediumSVM128_all = [];

targetsCoarseSVM128_all = [];
predsCoarseSVM128_all = [];

targets3KNN128_all = [];
preds3KNN128_all = [];

targets5KNN128_all = [];
preds5KNN128_all = [];

targets7KNN128_all = [];
preds7KNN128_all = [];

targets9KNN128_all = [];
preds9KNN128_all = [];

targets5Tree128_all = [];
preds5Tree128_all = [];

targets10Tree128_all = [];
preds10Tree128_all = [];

targets20Tree128_all = [];
preds20Tree128_all = [];

targets30Tree128_all = [];
preds30Tree128_all = [];

targets40Tree128_all = [];
preds40Tree128_all = [];

targets50Tree128_all = [];
preds50Tree128_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM128 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM128,~] = predict(classificationLinearSVM128,trainingData(test,1:end-1));
    targetLinSVM128 = trainingData(test,end);
    targetsLinSVM128_all = [targetsLinSVM128_all; squeeze(targetLinSVM128)];
    predsLinSVM128_all = [predsLinSVM128_all; squeeze(predsLinSVM128)];
    
    [~,scoresLinSVM128] = resubPredict(fitPosterior(classificationLinearSVM128));
    [xLinSVM128,yLinSVM128,~,aucLinSVM128] = perfcurve(trainingData(train,end),scoresLinSVM128(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM128.mat','classificationLinearSVM128','-v7.3'); % majority voting
    %save('classificationLinearSVM128_30.mat','classificationLinearSVM128','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM128_20.mat','classificationLinearSVM128','-v7.3'); % Threshold 20%
    save('classificationLinearSVM128_10.mat','classificationLinearSVM128','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM128 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM128,~] = predict(classificationQuadSVM128,trainingData(test,1:end-1));
    targetQuadSVM128 = trainingData(test,end);
    targetsQuadSVM128_all = [targetsQuadSVM128_all; squeeze(targetQuadSVM128)];
    predsQuadSVM128_all = [predsQuadSVM128_all; squeeze(predsQuadSVM128)];
    
    [~,scoresQuadSVM128] = resubPredict(fitPosterior(classificationQuadSVM128));
    [xQuadSVM128,yQuadSVM128,~,aucQuadSVM128] = perfcurve(trainingData(train,end),scoresQuadSVM128(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM128.mat','classificationQuadSVM128','-v7.3');
    %save('classificationQuadSVM128_30.mat','classificationQuadSVM128','-v7.3');
    %save('classificationQuadSVM128_20.mat','classificationQuadSVM128','-v7.3');
    save('classificationQuadSVM128_10.mat','classificationQuadSVM128','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM128 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM128,~] = predict(classificationCubicSVM128,trainingData(test,1:end-1));
    targetCubicSVM128 = trainingData(test,end);
    targetsCubicSVM128_all = [targetsCubicSVM128_all; squeeze(targetCubicSVM128)];
    predsCubicSVM128_all = [predsCubicSVM128_all; squeeze(predsCubicSVM128)];
    
    [~,scoresCubicSVM128] = resubPredict(fitPosterior(classificationCubicSVM128));
    [xCubicSVM128,yCubicSVM128,~,aucCubicSVM128] = perfcurve(trainingData(train,end),scoresCubicSVM128(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM128.mat','classificationCubicSVM128','-v7.3');
    %save('classificationCubicSVM128_30.mat','classificationCubicSVM128','-v7.3');
    %save('classificationCubicSVM128_20.mat','classificationCubicSVM128','-v7.3');
    save('classificationCubicSVM128_10.mat','classificationCubicSVM128','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM128 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM128,~] = predict(classificationFineSVM128,trainingData(test,1:end-1));
    targetsFineSVM128 = trainingData(test,end);
    targetsFineSVM128_all = [targetsFineSVM128_all; squeeze(targetsFineSVM128)];
    predsFineSVM128_all = [predsFineSVM128_all; squeeze(predsFineSVM128)];
    
    [~,scoresFineSVM128] = resubPredict(fitPosterior(classificationFineSVM128));
    [xFineSVM128,yFineSVM128,~,aucFineSVM128] = perfcurve(trainingData(train,end),scoresFineSVM128(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM128.mat','classificationFineSVM128','-v7.3');
    %save('classificationFineSVM128_30.mat','classificationFineSVM128','-v7.3');
    %save('classificationFineSVM128_20.mat','classificationFineSVM128','-v7.3');
    save('classificationFineSVM128_10.mat','classificationFineSVM128','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM128 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM128,~] = predict(classificationMediumSVM128,trainingData(test,1:end-1));
    targetsMediumSVM128 = trainingData(test,end);
    targetsMediumSVM128_all = [targetsMediumSVM128_all; squeeze(targetsMediumSVM128)];
    predsMediumSVM128_all = [predsMediumSVM128_all; squeeze(predsMediumSVM128)];
    
    [~,scoresMediumSVM128] = resubPredict(fitPosterior(classificationMediumSVM128));
    [xMediumSVM128,yMediumSVM128,~,aucMediumSVM128] = perfcurve(trainingData(train,end),scoresMediumSVM128(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM128.mat','classificationMediumSVM128','-v7.3');
    %save('classificationMediumSVM128_30.mat','classificationMediumSVM128','-v7.3');
    %save('classificationMediumSVM128_20.mat','classificationMediumSVM128','-v7.3');
    save('classificationMediumSVM128_10.mat','classificationMediumSVM128','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM128 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM128,~] = predict(classificationCoarseSVM128,trainingData(test,1:end-1));
    targetsCoarseSVM128 = trainingData(test,end);
    targetsCoarseSVM128_all = [targetsCoarseSVM128_all; squeeze(targetsCoarseSVM128)];
    predsCoarseSVM128_all = [predsCoarseSVM128_all; squeeze(predsCoarseSVM128)];
    
    [~,scoresCoarseSVM128] = resubPredict(fitPosterior(classificationCoarseSVM128));
    [xCoarseSVM128,yCoarseSVM128,~,aucCoarseSVM128] = perfcurve(trainingData(train,end),scoresCoarseSVM128(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM128.mat','classificationCoarseSVM128','-v7.3');
    %save('classificationCoarseSVM128_30.mat','classificationCoarseSVM128','-v7.3');
    %save('classificationCoarseSVM128_20.mat','classificationCoarseSVM128','-v7.3');
    save('classificationCoarseSVM128_10.mat','classificationCoarseSVM128','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN128 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN128,~] = predict(classification3KNN128,trainingData(test,1:end-1));
    targets3KNN128 = trainingData(test,end);
    targets3KNN128_all = [targets3KNN128_all; squeeze(targets3KNN128)];
    preds3KNN128_all = [preds3KNN128_all; squeeze(preds3KNN128)];
    
    [~,scores3KNN128] = resubPredict((classification3KNN128));
    [x3KNN128,y3KNN128,~,auc3KNN128] = perfcurve(trainingData(train,end),scores3KNN128(:,2),1);
    t7 = toc;
    
    %save('classification3KNN128.mat','classification3KNN128','-v7.3');
    %save('classification3KNN128_30.mat','classification3KNN128','-v7.3');
    %save('classification3KNN128_20.mat','classification3KNN128','-v7.3');
    save('classification3KNN128_10.mat','classification3KNN128','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN128 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN128,~] = predict(classification5KNN128,trainingData(test,1:end-1));
    targets5KNN128 = trainingData(test,end);
    targets5KNN128_all = [targets5KNN128_all; squeeze(targets5KNN128)];
    preds5KNN128_all = [preds5KNN128_all; squeeze(preds5KNN128)];
    
    [~,scores5KNN128] = resubPredict((classification5KNN128));
    [x5KNN128,y5KNN128,~,auc5KNN128] = perfcurve(trainingData(train,end),scores5KNN128(:,2),1);
    t8 = toc;

    %save('classification5KNN128.mat','classification5KNN128','-v7.3');
    %save('classification5KNN128_30.mat','classification5KNN128','-v7.3');
    %save('classification5KNN128_20.mat','classification5KNN128','-v7.3');
    save('classification5KNN128_10.mat','classification5KNN128','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN128 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN128,~] = predict(classification7KNN128,trainingData(test,1:end-1));
    targets7KNN128 = trainingData(test,end);
    targets7KNN128_all = [targets7KNN128_all; squeeze(targets7KNN128)];
    preds7KNN128_all = [preds7KNN128_all; squeeze(preds7KNN128)];
    
    [~,scores7KNN128] = resubPredict((classification7KNN128));
    [x7KNN128,y7KNN128,~,auc7KNN128] = perfcurve(trainingData(train,end),scores7KNN128(:,2),1);
    t9 = toc;
    
    %save('classification7KNN128.mat','classification7KNN128','-v7.3');
    %save('classification7KNN128_30.mat','classification7KNN128','-v7.3');
    %save('classification7KNN128_20.mat','classification7KNN128','-v7.3');
    save('classification7KNN128_10.mat','classification7KNN128','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN128 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN128,~] = predict(classification9KNN128,trainingData(test,1:end-1));
    targets9KNN128 = trainingData(test,end);
    targets9KNN128_all = [targets9KNN128_all; squeeze(targets9KNN128)];
    preds9KNN128_all = [preds9KNN128_all; squeeze(preds9KNN128)];
    
    [~,scores9KNN128] = resubPredict((classification9KNN128));
    [x9KNN128,y9KNN128,~,auc9KNN128] = perfcurve(trainingData(train,end),scores9KNN128(:,2),1);
    t10 = toc;
    
    %save('classification9KNN128.mat','classification9KNN128','-v7.3');
    %save('classification9KNN128_30.mat','classification9KNN128','-v7.3');
    %save('classification9KNN128_20.mat','classification9KNN128','-v7.3');
    save('classification9KNN128_10.mat','classification9KNN128','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree128 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree128,~] = predict(classification5Tree128,trainingData(test,1:end-1));
    targets5Tree128 = trainingData(test,end);
    targets5Tree128_all = [targets5Tree128_all; squeeze(targets5Tree128)];
    preds5Tree128_all = [preds5Tree128_all; squeeze(preds5Tree128)];
    
    [~,scores5Tree128] = resubPredict((classification5Tree128));
    [x5Tree128,y5Tree128,~,auc5Tree128] = perfcurve(trainingData(train,end),scores5Tree128(:,2),1);
    t11 = toc;
    
    %save('classification5Tree128.mat','classification5Tree128','-v7.3');
    %save('classification5Tree128_30.mat','classification5Tree128','-v7.3');
    %save('classification5Tree128_20.mat','classification5Tree128','-v7.3');
    save('classification5Tree128_10.mat','classification5Tree128','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree128 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree128,~] = predict(classification10Tree128,trainingData(test,1:end-1));
    targets10Tree128 = trainingData(test,end);
    targets10Tree128_all = [targets10Tree128_all; squeeze(targets10Tree128)];
    preds10Tree128_all = [preds10Tree128_all; squeeze(preds10Tree128)];
    
    [~,scores10Tree128] = resubPredict((classification10Tree128));
    [x10Tree128,y10Tree128,~,auc10Tree128] = perfcurve(trainingData(train,end),scores10Tree128(:,2),1);
    t12 = toc;
    
    %save('classification10Tree128.mat','classification10Tree128','-v7.3');
    %save('classification10Tree128_30.mat','classification10Tree128','-v7.3');
    %save('classification10Tree128_20.mat','classification10Tree128','-v7.3');
    save('classification10Tree128_10.mat','classification10Tree128','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree128 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree128,~] = predict(classification20Tree128,trainingData(test,1:end-1));
    targets20Tree128 = trainingData(test,end);
    targets20Tree128_all = [targets20Tree128_all; squeeze(targets20Tree128)];
    preds20Tree128_all = [preds20Tree128_all; squeeze(preds20Tree128)];
    [~,scores20Tree128] = resubPredict((classification20Tree128));
    
    [x20Tree128,y20Tree128,~,auc20Tree128] = perfcurve(trainingData(train,end),scores20Tree128(:,2),1);
    t13 = toc;
    
    %save('classification20Tree128.mat','classification20Tree128','-v7.3');
    %save('classification20Tree128_30.mat','classification20Tree128','-v7.3');
    %save('classification20Tree128_20.mat','classification20Tree128','-v7.3');
    save('classification20Tree128_10.mat','classification20Tree128','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree128 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree128,~] = predict(classification30Tree128,trainingData(test,1:end-1));
    targets30Tree128 = trainingData(test,end);
    targets30Tree128_all = [targets30Tree128_all; squeeze(targets30Tree128)];
    preds30Tree128_all = [preds30Tree128_all; squeeze(preds30Tree128)];
    
    [~,scores30Tree128] = resubPredict((classification30Tree128));
    [x30Tree128,y30Tree128,~,auc30Tree128] = perfcurve(trainingData(train,end),scores30Tree128(:,2),1);
    t14 = toc;
    
    %save('classification30Tree128.mat','classification30Tree128','-v7.3');
    %save('classification30Tree128_30.mat','classification30Tree128','-v7.3');
    %save('classification30Tree128_20.mat','classification30Tree128','-v7.3');
    save('classification30Tree128_10.mat','classification30Tree128','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree128 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree128,~] = predict(classification40Tree128,trainingData(test,1:end-1));
    targets40Tree128 = trainingData(test,end);
    targets40Tree128_all = [targets40Tree128_all; squeeze(targets40Tree128)];
    preds40Tree128_all = [preds40Tree128_all; squeeze(preds40Tree128)];
    
    [~,scores40Tree128] = resubPredict((classification40Tree128));
    [x40Tree128,y40Tree128,~,auc40Tree128] = perfcurve(trainingData(train,end),scores40Tree128(:,2),1);
    t15 = toc;
    
    %save('classification40Tree128.mat','classification40Tree128','-v7.3');
    %save('classification40Tree128_30.mat','classification40Tree128','-v7.3');
    %save('classification40Tree128_20.mat','classification40Tree128','-v7.3');
    save('classification40Tree128_10.mat','classification40Tree128','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree128 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree128,~] = predict(classification50Tree128,trainingData(test,1:end-1));
    targets50Tree128 = trainingData(test,end);
    targets50Tree128_all = [targets50Tree128_all; squeeze(targets50Tree128)];
    preds50Tree128_all = [preds50Tree128_all; squeeze(preds50Tree128)];
    
    [~,scores50Tree128] = resubPredict((classification50Tree128));
    [x50Tree128,y50Tree128,~,auc50Tree128] = perfcurve(trainingData(train,end),scores50Tree128(:,2),1);
    t16 = toc;
    
    %save('classification50Tree128.mat','classification50Tree128','-v7.3');
    %save('classification50Tree128_30.mat','classification50Tree128','-v7.3');
    %save('classification50Tree128_20.mat','classification50Tree128','-v7.3');
    save('classification50Tree128_10.mat','classification50Tree128','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM128_all,predsLinSVM128_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM128_all,predsQuadSVM128_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM128_all,predsCubicSVM128_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM128_all,predsFineSVM128_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM128_all,predsMediumSVM128_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM128_all,predsCoarseSVM128_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN128_all,preds3KNN128_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN128_all,preds5KNN128_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN128_all,preds7KNN128_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN128_all,preds9KNN128_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree128_all,preds5Tree128_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree128_all,preds10Tree128_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree128_all,preds20Tree128_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree128_all,preds30Tree128_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree128_all,preds40Tree128_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree128_all,preds50Tree128_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM128,yLinSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM128,yQuadSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM128,yCubicSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM128,yFineSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM128,yMediumSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM128,yCoarseSVM128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN128,y3KNN128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN128,y5KNN128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN128,y7KNN128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN128,y9KNN128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree128,y5Tree128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree128,y10Tree128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree128,y20Tree128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree128,y30Tree128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree128,y40Tree128,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree128,y50Tree128,'LineWidth',2)
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
AUC = [aucLinSVM128;aucQuadSVM128;aucCubicSVM128;aucFineSVM128;aucMediumSVM128;...
    aucCoarseSVM128;auc3KNN128;auc5KNN128;auc7KNN128;auc9KNN128;auc5Tree128;...
    auc10Tree128;auc20Tree128;auc30Tree128;auc40Tree128;auc50Tree128];

T = table(AUC,Time,Time_total);

