function T = trainClassifier10(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM10_all = [];
predsLinSVM10_all = [];

targetsQuadSVM10_all = [];
predsQuadSVM10_all = [];

targetsCubicSVM10_all = [];
predsCubicSVM10_all = [];

targetsFineSVM10_all = [];
predsFineSVM10_all = [];

targetsMediumSVM10_all = [];
predsMediumSVM10_all = [];

targetsCoarseSVM10_all = [];
predsCoarseSVM10_all = [];

targets3KNN10_all = [];
preds3KNN10_all = [];

targets5KNN10_all = [];
preds5KNN10_all = [];

targets7KNN10_all = [];
preds7KNN10_all = [];

targets9KNN10_all = [];
preds9KNN10_all = [];

targets5Tree10_all = [];
preds5Tree10_all = [];

targets10Tree10_all = [];
preds10Tree10_all = [];

targets20Tree10_all = [];
preds20Tree10_all = [];

targets30Tree10_all = [];
preds30Tree10_all = [];

targets40Tree10_all = [];
preds40Tree10_all = [];

targets50Tree10_all = [];
preds50Tree10_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM10 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM10,~] = predict(classificationLinearSVM10,trainingData(test,1:end-1));
    targetLinSVM10 = trainingData(test,end);
    targetsLinSVM10_all = [targetsLinSVM10_all; squeeze(targetLinSVM10)];
    predsLinSVM10_all = [predsLinSVM10_all; squeeze(predsLinSVM10)];
    
    [~,scoresLinSVM10] = resubPredict(fitPosterior(classificationLinearSVM10));
    [xLinSVM10,yLinSVM10,~,aucLinSVM10] = perfcurve(trainingData(train,end),scoresLinSVM10(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM10.mat','classificationLinearSVM10','-v7.3'); % majority voting
    %save('classificationLinearSVM10_30.mat','classificationLinearSVM10','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM10_20.mat','classificationLinearSVM10','-v7.3'); % Threshold 20%
    save('classificationLinearSVM10_10.mat','classificationLinearSVM10','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM10 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM10,~] = predict(classificationQuadSVM10,trainingData(test,1:end-1));
    targetQuadSVM10 = trainingData(test,end);
    targetsQuadSVM10_all = [targetsQuadSVM10_all; squeeze(targetQuadSVM10)];
    predsQuadSVM10_all = [predsQuadSVM10_all; squeeze(predsQuadSVM10)];
    
    [~,scoresQuadSVM10] = resubPredict(fitPosterior(classificationQuadSVM10));
    [xQuadSVM10,yQuadSVM10,~,aucQuadSVM10] = perfcurve(trainingData(train,end),scoresQuadSVM10(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM10.mat','classificationQuadSVM10','-v7.3');
    %save('classificationQuadSVM10_30.mat','classificationQuadSVM10','-v7.3');
    %save('classificationQuadSVM10_20.mat','classificationQuadSVM10','-v7.3');
    save('classificationQuadSVM10_10.mat','classificationQuadSVM10','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM10 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM10,~] = predict(classificationCubicSVM10,trainingData(test,1:end-1));
    targetCubicSVM10 = trainingData(test,end);
    targetsCubicSVM10_all = [targetsCubicSVM10_all; squeeze(targetCubicSVM10)];
    predsCubicSVM10_all = [predsCubicSVM10_all; squeeze(predsCubicSVM10)];
    
    [~,scoresCubicSVM10] = resubPredict(fitPosterior(classificationCubicSVM10));
    [xCubicSVM10,yCubicSVM10,~,aucCubicSVM10] = perfcurve(trainingData(train,end),scoresCubicSVM10(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM10.mat','classificationCubicSVM10','-v7.3');
    %save('classificationCubicSVM10_30.mat','classificationCubicSVM10','-v7.3');
    %save('classificationCubicSVM10_20.mat','classificationCubicSVM10','-v7.3');
    save('classificationCubicSVM10_10.mat','classificationCubicSVM10','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM10 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM10,~] = predict(classificationFineSVM10,trainingData(test,1:end-1));
    targetsFineSVM10 = trainingData(test,end);
    targetsFineSVM10_all = [targetsFineSVM10_all; squeeze(targetsFineSVM10)];
    predsFineSVM10_all = [predsFineSVM10_all; squeeze(predsFineSVM10)];
    
    [~,scoresFineSVM10] = resubPredict(fitPosterior(classificationFineSVM10));
    [xFineSVM10,yFineSVM10,~,aucFineSVM10] = perfcurve(trainingData(train,end),scoresFineSVM10(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM10.mat','classificationFineSVM10','-v7.3');
    %save('classificationFineSVM10_30.mat','classificationFineSVM10','-v7.3');
    %save('classificationFineSVM10_20.mat','classificationFineSVM10','-v7.3');
    save('classificationFineSVM10_10.mat','classificationFineSVM10','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM10 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM10,~] = predict(classificationMediumSVM10,trainingData(test,1:end-1));
    targetsMediumSVM10 = trainingData(test,end);
    targetsMediumSVM10_all = [targetsMediumSVM10_all; squeeze(targetsMediumSVM10)];
    predsMediumSVM10_all = [predsMediumSVM10_all; squeeze(predsMediumSVM10)];
    
    [~,scoresMediumSVM10] = resubPredict(fitPosterior(classificationMediumSVM10));
    [xMediumSVM10,yMediumSVM10,~,aucMediumSVM10] = perfcurve(trainingData(train,end),scoresMediumSVM10(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM10.mat','classificationMediumSVM10','-v7.3');
    %save('classificationMediumSVM10_30.mat','classificationMediumSVM10','-v7.3');
    %save('classificationMediumSVM10_20.mat','classificationMediumSVM10','-v7.3');
    save('classificationMediumSVM10_10.mat','classificationMediumSVM10','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM10 = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM10,~] = predict(classificationCoarseSVM10,trainingData(test,1:end-1));
    targetsCoarseSVM10 = trainingData(test,end);
    targetsCoarseSVM10_all = [targetsCoarseSVM10_all; squeeze(targetsCoarseSVM10)];
    predsCoarseSVM10_all = [predsCoarseSVM10_all; squeeze(predsCoarseSVM10)];
    
    [~,scoresCoarseSVM10] = resubPredict(fitPosterior(classificationCoarseSVM10));
    [xCoarseSVM10,yCoarseSVM10,~,aucCoarseSVM10] = perfcurve(trainingData(train,end),scoresCoarseSVM10(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM10.mat','classificationCoarseSVM10','-v7.3');
    %save('classificationCoarseSVM10_30.mat','classificationCoarseSVM10','-v7.3');
    %save('classificationCoarseSVM10_20.mat','classificationCoarseSVM10','-v7.3');
    save('classificationCoarseSVM10_10.mat','classificationCoarseSVM10','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN10 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN10,~] = predict(classification3KNN10,trainingData(test,1:end-1));
    targets3KNN10 = trainingData(test,end);
    targets3KNN10_all = [targets3KNN10_all; squeeze(targets3KNN10)];
    preds3KNN10_all = [preds3KNN10_all; squeeze(preds3KNN10)];
    
    [~,scores3KNN10] = resubPredict((classification3KNN10));
    [x3KNN10,y3KNN10,~,auc3KNN10] = perfcurve(trainingData(train,end),scores3KNN10(:,2),1);
    t7 = toc;
    
    %save('classification3KNN10.mat','classification3KNN10','-v7.3');
    %save('classification3KNN10_30.mat','classification3KNN10','-v7.3');
    %save('classification3KNN10_20.mat','classification3KNN10','-v7.3');
    save('classification3KNN10_10.mat','classification3KNN10','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN10 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN10,~] = predict(classification5KNN10,trainingData(test,1:end-1));
    targets5KNN10 = trainingData(test,end);
    targets5KNN10_all = [targets5KNN10_all; squeeze(targets5KNN10)];
    preds5KNN10_all = [preds5KNN10_all; squeeze(preds5KNN10)];
    
    [~,scores5KNN10] = resubPredict((classification5KNN10));
    [x5KNN10,y5KNN10,~,auc5KNN10] = perfcurve(trainingData(train,end),scores5KNN10(:,2),1);
    t8 = toc;

    %save('classification5KNN10.mat','classification5KNN10','-v7.3');
    %save('classification5KNN10_30.mat','classification5KNN10','-v7.3');
    %save('classification5KNN10_20.mat','classification5KNN10','-v7.3');
    save('classification5KNN10_10.mat','classification5KNN10','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN10 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN10,~] = predict(classification7KNN10,trainingData(test,1:end-1));
    targets7KNN10 = trainingData(test,end);
    targets7KNN10_all = [targets7KNN10_all; squeeze(targets7KNN10)];
    preds7KNN10_all = [preds7KNN10_all; squeeze(preds7KNN10)];
    
    [~,scores7KNN10] = resubPredict((classification7KNN10));
    [x7KNN10,y7KNN10,~,auc7KNN10] = perfcurve(trainingData(train,end),scores7KNN10(:,2),1);
    t9 = toc;
    
    %save('classification7KNN10.mat','classification7KNN10','-v7.3');
    %save('classification7KNN10_30.mat','classification7KNN10','-v7.3');
    %save('classification7KNN10_20.mat','classification7KNN10','-v7.3');
    save('classification7KNN10_10.mat','classification7KNN10','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN10 = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN10,~] = predict(classification9KNN10,trainingData(test,1:end-1));
    targets9KNN10 = trainingData(test,end);
    targets9KNN10_all = [targets9KNN10_all; squeeze(targets9KNN10)];
    preds9KNN10_all = [preds9KNN10_all; squeeze(preds9KNN10)];
    
    [~,scores9KNN10] = resubPredict((classification9KNN10));
    [x9KNN10,y9KNN10,~,auc9KNN10] = perfcurve(trainingData(train,end),scores9KNN10(:,2),1);
    t10 = toc;
    
    %save('classification9KNN10.mat','classification9KNN10','-v7.3');
    %save('classification9KNN10_30.mat','classification9KNN10','-v7.3');
    %save('classification9KNN10_20.mat','classification9KNN10','-v7.3');
    save('classification9KNN10_10.mat','classification9KNN10','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree10 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree10,~] = predict(classification5Tree10,trainingData(test,1:end-1));
    targets5Tree10 = trainingData(test,end);
    targets5Tree10_all = [targets5Tree10_all; squeeze(targets5Tree10)];
    preds5Tree10_all = [preds5Tree10_all; squeeze(preds5Tree10)];
    
    [~,scores5Tree10] = resubPredict((classification5Tree10));
    [x5Tree10,y5Tree10,~,auc5Tree10] = perfcurve(trainingData(train,end),scores5Tree10(:,2),1);
    t11 = toc;
    
    %save('classification5Tree10.mat','classification5Tree10','-v7.3');
    %save('classification5Tree10_30.mat','classification5Tree10','-v7.3');
    %save('classification5Tree10_20.mat','classification5Tree10','-v7.3');
    save('classification5Tree10_10.mat','classification5Tree10','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree10 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree10,~] = predict(classification10Tree10,trainingData(test,1:end-1));
    targets10Tree10 = trainingData(test,end);
    targets10Tree10_all = [targets10Tree10_all; squeeze(targets10Tree10)];
    preds10Tree10_all = [preds10Tree10_all; squeeze(preds10Tree10)];
    
    [~,scores10Tree10] = resubPredict((classification10Tree10));
    [x10Tree10,y10Tree10,~,auc10Tree10] = perfcurve(trainingData(train,end),scores10Tree10(:,2),1);
    t12 = toc;
    
    %save('classification10Tree10.mat','classification10Tree10','-v7.3');
    %save('classification10Tree10_30.mat','classification10Tree10','-v7.3');
    %save('classification10Tree10_20.mat','classification10Tree10','-v7.3');
    save('classification10Tree10_10.mat','classification10Tree10','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree10 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree10,~] = predict(classification20Tree10,trainingData(test,1:end-1));
    targets20Tree10 = trainingData(test,end);
    targets20Tree10_all = [targets20Tree10_all; squeeze(targets20Tree10)];
    preds20Tree10_all = [preds20Tree10_all; squeeze(preds20Tree10)];
    [~,scores20Tree10] = resubPredict((classification20Tree10));
    
    [x20Tree10,y20Tree10,~,auc20Tree10] = perfcurve(trainingData(train,end),scores20Tree10(:,2),1);
    t13 = toc;
    
    %save('classification20Tree10.mat','classification20Tree10','-v7.3');
    %save('classification20Tree10_30.mat','classification20Tree10','-v7.3');
    %save('classification20Tree10_20.mat','classification20Tree10','-v7.3');
    save('classification20Tree10_10.mat','classification20Tree10','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree10 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree10,~] = predict(classification30Tree10,trainingData(test,1:end-1));
    targets30Tree10 = trainingData(test,end);
    targets30Tree10_all = [targets30Tree10_all; squeeze(targets30Tree10)];
    preds30Tree10_all = [preds30Tree10_all; squeeze(preds30Tree10)];
    
    [~,scores30Tree10] = resubPredict((classification30Tree10));
    [x30Tree10,y30Tree10,~,auc30Tree10] = perfcurve(trainingData(train,end),scores30Tree10(:,2),1);
    t14 = toc;
    
    %save('classification30Tree10.mat','classification30Tree10','-v7.3');
    %save('classification30Tree10_30.mat','classification30Tree10','-v7.3');
    %save('classification30Tree10_20.mat','classification30Tree10','-v7.3');
    save('classification30Tree10_10.mat','classification30Tree10','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree10 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree10,~] = predict(classification40Tree10,trainingData(test,1:end-1));
    targets40Tree10 = trainingData(test,end);
    targets40Tree10_all = [targets40Tree10_all; squeeze(targets40Tree10)];
    preds40Tree10_all = [preds40Tree10_all; squeeze(preds40Tree10)];
    
    [~,scores40Tree10] = resubPredict((classification40Tree10));
    [x40Tree10,y40Tree10,~,auc40Tree10] = perfcurve(trainingData(train,end),scores40Tree10(:,2),1);
    t15 = toc;
    
    %save('classification40Tree10.mat','classification40Tree10','-v7.3');
    %save('classification40Tree10_30.mat','classification40Tree10','-v7.3');
    %save('classification40Tree10_20.mat','classification40Tree10','-v7.3');
    save('classification40Tree10_10.mat','classification40Tree10','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree10 = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree10,~] = predict(classification50Tree10,trainingData(test,1:end-1));
    targets50Tree10 = trainingData(test,end);
    targets50Tree10_all = [targets50Tree10_all; squeeze(targets50Tree10)];
    preds50Tree10_all = [preds50Tree10_all; squeeze(preds50Tree10)];
    
    [~,scores50Tree10] = resubPredict((classification50Tree10));
    [x50Tree10,y50Tree10,~,auc50Tree10] = perfcurve(trainingData(train,end),scores50Tree10(:,2),1);
    t16 = toc;
    
    %save('classification50Tree10.mat','classification50Tree10','-v7.3');
    %save('classification50Tree10_30.mat','classification50Tree10','-v7.3');
    %save('classification50Tree10_20.mat','classification50Tree10','-v7.3');
    save('classification50Tree10_10.mat','classification50Tree10','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM10_all,predsLinSVM10_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM10_all,predsQuadSVM10_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM10_all,predsCubicSVM10_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM10_all,predsFineSVM10_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM10_all,predsMediumSVM10_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM10_all,predsCoarseSVM10_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN10_all,preds3KNN10_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN10_all,preds5KNN10_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN10_all,preds7KNN10_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN10_all,preds9KNN10_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree10_all,preds5Tree10_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree10_all,preds10Tree10_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree10_all,preds20Tree10_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree10_all,preds30Tree10_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree10_all,preds40Tree10_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree10_all,preds50Tree10_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM10,yLinSVM10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM10,yQuadSVM10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM10,yCubicSVM10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM10,yFineSVM10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM10,yMediumSVM10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM10,yCoarseSVM10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN10,y3KNN10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN10,y5KNN10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN10,y7KNN10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN10,y9KNN10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree10,y5Tree10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree10,y10Tree10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree10,y20Tree10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree10,y30Tree10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree10,y40Tree10,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree10,y50Tree10,'LineWidth',2)
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
AUC = [aucLinSVM10;aucQuadSVM10;aucCubicSVM10;aucFineSVM10;aucMediumSVM10;...
    aucCoarseSVM10;auc3KNN10;auc5KNN10;auc7KNN10;auc9KNN10;auc5Tree10;...
    auc10Tree10;auc20Tree10;auc30Tree10;auc40Tree10;auc50Tree10];

T = table(AUC,Time,Time_total);

