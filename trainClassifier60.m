function T = trainClassifier60(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM60_all = [];
predsLinSVM60_all = [];

targetsQuadSVM60_all = [];
predsQuadSVM60_all = [];

targetsCubicSVM60_all = [];
predsCubicSVM60_all = [];

targetsFineSVM60_all = [];
predsFineSVM60_all = [];

targetsMediumSVM60_all = [];
predsMediumSVM60_all = [];

targetsCoarseSVM60_all = [];
predsCoarseSVM60_all = [];

targets3KNN60_all = [];
preds3KNN60_all = [];

targets5KNN60_all = [];
preds5KNN60_all = [];

targets7KNN60_all = [];
preds7KNN60_all = [];

targets9KNN60_all = [];
preds9KNN60_all = [];

targets5Tree60_all = [];
preds5Tree60_all = [];

targets10Tree60_all = [];
preds10Tree60_all = [];

targets20Tree60_all = [];
preds20Tree60_all = [];

targets30Tree60_all = [];
preds30Tree60_all = [];

targets40Tree60_all = [];
preds40Tree60_all = [];

targets50Tree60_all = [];
preds50Tree60_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM60 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM60,~] = predict(classificationLinearSVM60,trainingData(test,1:end-1));
    targetLinSVM60 = trainingData(test,end);
    targetsLinSVM60_all = [targetsLinSVM60_all; squeeze(targetLinSVM60)];
    predsLinSVM60_all = [predsLinSVM60_all; squeeze(predsLinSVM60)];
    
    [~,scoresLinSVM60] = resubPredict(fitPosterior(classificationLinearSVM60));
    [xLinSVM60,yLinSVM60,~,aucLinSVM60] = perfcurve(trainingData(train,end),scoresLinSVM60(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM60.mat','classificationLinearSVM60','-v7.3'); % majority voting
    save('classificationLinearSVM60_30.mat','classificationLinearSVM60','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM60_20.mat','classificationLinearSVM60','-v7.3'); % Threshold 20%
    %save('classificationLinearSVM60_10.mat','classificationLinearSVM60','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM60 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM60,~] = predict(classificationQuadSVM60,trainingData(test,1:end-1));
    targetQuadSVM60 = trainingData(test,end);
    targetsQuadSVM60_all = [targetsQuadSVM60_all; squeeze(targetQuadSVM60)];
    predsQuadSVM60_all = [predsQuadSVM60_all; squeeze(predsQuadSVM60)];
    
    [~,scoresQuadSVM60] = resubPredict(fitPosterior(classificationQuadSVM60));
    [xQuadSVM60,yQuadSVM60,~,aucQuadSVM60] = perfcurve(trainingData(train,end),scoresQuadSVM60(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM60.mat','classificationQuadSVM60','-v7.3');
    save('classificationQuadSVM60_30.mat','classificationQuadSVM60','-v7.3');
    %save('classificationQuadSVM60_20.mat','classificationQuadSVM60','-v7.3');
    %save('classificationQuadSVM60_10.mat','classificationQuadSVM60','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM60 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM60,~] = predict(classificationCubicSVM60,trainingData(test,1:end-1));
    targetCubicSVM60 = trainingData(test,end);
    targetsCubicSVM60_all = [targetsCubicSVM60_all; squeeze(targetCubicSVM60)];
    predsCubicSVM60_all = [predsCubicSVM60_all; squeeze(predsCubicSVM60)];
    
    [~,scoresCubicSVM60] = resubPredict(fitPosterior(classificationCubicSVM60));
    [xCubicSVM60,yCubicSVM60,~,aucCubicSVM60] = perfcurve(trainingData(train,end),scoresCubicSVM60(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM60.mat','classificationCubicSVM60','-v7.3');
    save('classificationCubicSVM60_30.mat','classificationCubicSVM60','-v7.3');
    %save('classificationCubicSVM60_20.mat','classificationCubicSVM60','-v7.3');
    %save('classificationCubicSVM60_10.mat','classificationCubicSVM60','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM60 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM60,~] = predict(classificationFineSVM60,trainingData(test,1:end-1));
    targetsFineSVM60 = trainingData(test,end);
    targetsFineSVM60_all = [targetsFineSVM60_all; squeeze(targetsFineSVM60)];
    predsFineSVM60_all = [predsFineSVM60_all; squeeze(predsFineSVM60)];
    
    [~,scoresFineSVM60] = resubPredict(fitPosterior(classificationFineSVM60));
    [xFineSVM60,yFineSVM60,~,aucFineSVM60] = perfcurve(trainingData(train,end),scoresFineSVM60(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM60.mat','classificationFineSVM60','-v7.3');
    save('classificationFineSVM60_30.mat','classificationFineSVM60','-v7.3');
    %save('classificationFineSVM60_20.mat','classificationFineSVM60','-v7.3');
    %save('classificationFineSVM60_10.mat','classificationFineSVM60','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM60 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM60,~] = predict(classificationMediumSVM60,trainingData(test,1:end-1));
    targetsMediumSVM60 = trainingData(test,end);
    targetsMediumSVM60_all = [targetsMediumSVM60_all; squeeze(targetsMediumSVM60)];
    predsMediumSVM60_all = [predsMediumSVM60_all; squeeze(predsMediumSVM60)];
    
    [~,scoresMediumSVM60] = resubPredict(fitPosterior(classificationMediumSVM60));
    [xMediumSVM60,yMediumSVM60,~,aucMediumSVM60] = perfcurve(trainingData(train,end),scoresMediumSVM60(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM60.mat','classificationMediumSVM60','-v7.3');
    save('classificationMediumSVM60_30.mat','classificationMediumSVM60','-v7.3');
    %save('classificationMediumSVM60_20.mat','classificationMediumSVM60','-v7.3');
    %save('classificationMediumSVM60_10.mat','classificationMediumSVM60','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM60 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM60,~] = predict(classificationCoarseSVM60,trainingData(test,1:end-1));
    targetsCoarseSVM60 = trainingData(test,end);
    targetsCoarseSVM60_all = [targetsCoarseSVM60_all; squeeze(targetsCoarseSVM60)];
    predsCoarseSVM60_all = [predsCoarseSVM60_all; squeeze(predsCoarseSVM60)];
    
    [~,scoresCoarseSVM60] = resubPredict(fitPosterior(classificationCoarseSVM60));
    [xCoarseSVM60,yCoarseSVM60,~,aucCoarseSVM60] = perfcurve(trainingData(train,end),scoresCoarseSVM60(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM60.mat','classificationCoarseSVM60','-v7.3');
    save('classificationCoarseSVM60_30.mat','classificationCoarseSVM60','-v7.3');
    %save('classificationCoarseSVM60_20.mat','classificationCoarseSVM60','-v7.3');
    %save('classificationCoarseSVM60_10.mat','classificationCoarseSVM60','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN60 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN60,~] = predict(classification3KNN60,trainingData(test,1:end-1));
    targets3KNN60 = trainingData(test,end);
    targets3KNN60_all = [targets3KNN60_all; squeeze(targets3KNN60)];
    preds3KNN60_all = [preds3KNN60_all; squeeze(preds3KNN60)];
    
    [~,scores3KNN60] = resubPredict((classification3KNN60));
    [x3KNN60,y3KNN60,~,auc3KNN60] = perfcurve(trainingData(train,end),scores3KNN60(:,2),1);
    t7 = toc;
    
    %save('classification3KNN60.mat','classification3KNN60','-v7.3');
    save('classification3KNN60_30.mat','classification3KNN60','-v7.3');
    %save('classification3KNN60_20.mat','classification3KNN60','-v7.3');
    %save('classification3KNN60_10.mat','classification3KNN60','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN60 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN60,~] = predict(classification5KNN60,trainingData(test,1:end-1));
    targets5KNN60 = trainingData(test,end);
    targets5KNN60_all = [targets5KNN60_all; squeeze(targets5KNN60)];
    preds5KNN60_all = [preds5KNN60_all; squeeze(preds5KNN60)];
    
    [~,scores5KNN60] = resubPredict((classification5KNN60));
    [x5KNN60,y5KNN60,~,auc5KNN60] = perfcurve(trainingData(train,end),scores5KNN60(:,2),1);
    t8 = toc;

    %save('classification5KNN60.mat','classification5KNN60','-v7.3');
    save('classification5KNN60_30.mat','classification5KNN60','-v7.3');
    %save('classification5KNN60_20.mat','classification5KNN60','-v7.3');
    %save('classification5KNN60_10.mat','classification5KNN60','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN60 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN60,~] = predict(classification7KNN60,trainingData(test,1:end-1));
    targets7KNN60 = trainingData(test,end);
    targets7KNN60_all = [targets7KNN60_all; squeeze(targets7KNN60)];
    preds7KNN60_all = [preds7KNN60_all; squeeze(preds7KNN60)];
    
    [~,scores7KNN60] = resubPredict((classification7KNN60));
    [x7KNN60,y7KNN60,~,auc7KNN60] = perfcurve(trainingData(train,end),scores7KNN60(:,2),1);
    t9 = toc;
    
    %save('classification7KNN60.mat','classification7KNN60','-v7.3');
    save('classification7KNN60_30.mat','classification7KNN60','-v7.3');
    %save('classification7KNN60_20.mat','classification7KNN60','-v7.3');
    %save('classification7KNN60_10.mat','classification7KNN60','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN60 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN60,~] = predict(classification9KNN60,trainingData(test,1:end-1));
    targets9KNN60 = trainingData(test,end);
    targets9KNN60_all = [targets9KNN60_all; squeeze(targets9KNN60)];
    preds9KNN60_all = [preds9KNN60_all; squeeze(preds9KNN60)];
    
    [~,scores9KNN60] = resubPredict((classification9KNN60));
    [x9KNN60,y9KNN60,~,auc9KNN60] = perfcurve(trainingData(train,end),scores9KNN60(:,2),1);
    t10 = toc;
    
    %save('classification9KNN60.mat','classification9KNN60','-v7.3');
    save('classification9KNN60_30.mat','classification9KNN60','-v7.3');
    %save('classification9KNN60_20.mat','classification9KNN60','-v7.3');
    %save('classification9KNN60_10.mat','classification9KNN60','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree60 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree60,~] = predict(classification5Tree60,trainingData(test,1:end-1));
    targets5Tree60 = trainingData(test,end);
    targets5Tree60_all = [targets5Tree60_all; squeeze(targets5Tree60)];
    preds5Tree60_all = [preds5Tree60_all; squeeze(preds5Tree60)];
    
    [~,scores5Tree60] = resubPredict((classification5Tree60));
    [x5Tree60,y5Tree60,~,auc5Tree60] = perfcurve(trainingData(train,end),scores5Tree60(:,2),1);
    t11 = toc;
    
    %save('classification5Tree60.mat','classification5Tree60','-v7.3');
    save('classification5Tree60_30.mat','classification5Tree60','-v7.3');
    %save('classification5Tree60_20.mat','classification5Tree60','-v7.3');
    %save('classification5Tree60_10.mat','classification5Tree60','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree60 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree60,~] = predict(classification10Tree60,trainingData(test,1:end-1));
    targets10Tree60 = trainingData(test,end);
    targets10Tree60_all = [targets10Tree60_all; squeeze(targets10Tree60)];
    preds10Tree60_all = [preds10Tree60_all; squeeze(preds10Tree60)];
    
    [~,scores10Tree60] = resubPredict((classification10Tree60));
    [x10Tree60,y10Tree60,~,auc10Tree60] = perfcurve(trainingData(train,end),scores10Tree60(:,2),1);
    t12 = toc;
    
    %save('classification10Tree60.mat','classification10Tree60','-v7.3');
    save('classification10Tree60_30.mat','classification10Tree60','-v7.3');
    %save('classification10Tree60_20.mat','classification10Tree60','-v7.3');
    %save('classification10Tree60_10.mat','classification10Tree60','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree60 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree60,~] = predict(classification20Tree60,trainingData(test,1:end-1));
    targets20Tree60 = trainingData(test,end);
    targets20Tree60_all = [targets20Tree60_all; squeeze(targets20Tree60)];
    preds20Tree60_all = [preds20Tree60_all; squeeze(preds20Tree60)];
    [~,scores20Tree60] = resubPredict((classification20Tree60));
    
    [x20Tree60,y20Tree60,~,auc20Tree60] = perfcurve(trainingData(train,end),scores20Tree60(:,2),1);
    t13 = toc;
    
    %save('classification20Tree60.mat','classification20Tree60','-v7.3');
    save('classification20Tree60_30.mat','classification20Tree60','-v7.3');
    %save('classification20Tree60_20.mat','classification20Tree60','-v7.3');
    %save('classification20Tree60_10.mat','classification20Tree60','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree60 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree60,~] = predict(classification30Tree60,trainingData(test,1:end-1));
    targets30Tree60 = trainingData(test,end);
    targets30Tree60_all = [targets30Tree60_all; squeeze(targets30Tree60)];
    preds30Tree60_all = [preds30Tree60_all; squeeze(preds30Tree60)];
    
    [~,scores30Tree60] = resubPredict((classification30Tree60));
    [x30Tree60,y30Tree60,~,auc30Tree60] = perfcurve(trainingData(train,end),scores30Tree60(:,2),1);
    t14 = toc;
    
    %save('classification30Tree60.mat','classification30Tree60','-v7.3');
    save('classification30Tree60_30.mat','classification30Tree60','-v7.3');
    %save('classification30Tree60_20.mat','classification30Tree60','-v7.3');
    %save('classification30Tree60_10.mat','classification30Tree60','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree60 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree60,~] = predict(classification40Tree60,trainingData(test,1:end-1));
    targets40Tree60 = trainingData(test,end);
    targets40Tree60_all = [targets40Tree60_all; squeeze(targets40Tree60)];
    preds40Tree60_all = [preds40Tree60_all; squeeze(preds40Tree60)];
    
    [~,scores40Tree60] = resubPredict((classification40Tree60));
    [x40Tree60,y40Tree60,~,auc40Tree60] = perfcurve(trainingData(train,end),scores40Tree60(:,2),1);
    t15 = toc;
    
    %save('classification40Tree60.mat','classification40Tree60','-v7.3');
    save('classification40Tree60_30.mat','classification40Tree60','-v7.3');
    %save('classification40Tree60_20.mat','classification40Tree60','-v7.3');
    %save('classification40Tree60_10.mat','classification40Tree60','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree60 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree60,~] = predict(classification50Tree60,trainingData(test,1:end-1));
    targets50Tree60 = trainingData(test,end);
    targets50Tree60_all = [targets50Tree60_all; squeeze(targets50Tree60)];
    preds50Tree60_all = [preds50Tree60_all; squeeze(preds50Tree60)];
    
    [~,scores50Tree60] = resubPredict((classification50Tree60));
    [x50Tree60,y50Tree60,~,auc50Tree60] = perfcurve(trainingData(train,end),scores50Tree60(:,2),1);
    t16 = toc;
    
    %save('classification50Tree60.mat','classification50Tree60','-v7.3');
    save('classification50Tree60_30.mat','classification50Tree60','-v7.3');
    %save('classification50Tree60_20.mat','classification50Tree60','-v7.3');
    %save('classification50Tree60_10.mat','classification50Tree60','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM60_all,predsLinSVM60_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM60_all,predsQuadSVM60_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM60_all,predsCubicSVM60_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM60_all,predsFineSVM60_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM60_all,predsMediumSVM60_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM60_all,predsCoarseSVM60_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN60_all,preds3KNN60_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN60_all,preds5KNN60_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN60_all,preds7KNN60_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN60_all,preds9KNN60_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree60_all,preds5Tree60_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree60_all,preds10Tree60_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree60_all,preds20Tree60_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree60_all,preds30Tree60_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree60_all,preds40Tree60_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree60_all,preds50Tree60_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM60,yLinSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM60,yQuadSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM60,yCubicSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM60,yFineSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM60,yMediumSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM60,yCoarseSVM60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN60,y3KNN60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN60,y5KNN60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN60,y7KNN60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN60,y9KNN60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree60,y5Tree60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree60,y10Tree60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree60,y20Tree60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree60,y30Tree60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree60,y40Tree60,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree60,y50Tree60,'LineWidth',2)
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
AUC = [aucLinSVM60;aucQuadSVM60;aucCubicSVM60;aucFineSVM60;aucMediumSVM60;...
    aucCoarseSVM60;auc3KNN60;auc5KNN60;auc7KNN60;auc9KNN60;auc5Tree60;...
    auc10Tree60;auc20Tree60;auc30Tree60;auc40Tree60;auc50Tree60];

T = table(AUC,Time,Time_total);

