function T = trainClassifier128ADASYN(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM128ADASYN_all = [];
predsLinSVM128ADASYN_all = [];

targetsQuadSVM128ADASYN_all = [];
predsQuadSVM128ADASYN_all = [];

targetsCubicSVM128ADASYN_all = [];
predsCubicSVM128ADASYN_all = [];

targetsFineSVM128ADASYN_all = [];
predsFineSVM128ADASYN_all = [];

targetsMediumSVM128ADASYN_all = [];
predsMediumSVM128ADASYN_all = [];

targetsCoarseSVM128ADASYN_all = [];
predsCoarseSVM128ADASYN_all = [];

targets3KNN128ADASYN_all = [];
preds3KNN128ADASYN_all = [];

targets5KNN128ADASYN_all = [];
preds5KNN128ADASYN_all = [];

targets7KNN128ADASYN_all = [];
preds7KNN128ADASYN_all = [];

targets9KNN128ADASYN_all = [];
preds9KNN128ADASYN_all = [];

targets5Tree128ADASYN_all = [];
preds5Tree128ADASYN_all = [];

targets10Tree128ADASYN_all = [];
preds10Tree128ADASYN_all = [];

targets20Tree128ADASYN_all = [];
preds20Tree128ADASYN_all = [];

targets30Tree128ADASYN_all = [];
preds30Tree128ADASYN_all = [];

targets40Tree128ADASYN_all = [];
preds40Tree128ADASYN_all = [];

targets50Tree128ADASYN_all = [];
preds50Tree128ADASYN_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM128ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM128ADASYN,~] = predict(classificationLinearSVM128ADASYN,trainingData(test,1:end-1));
    targetLinSVM128ADASYN = trainingData(test,end);
    targetsLinSVM128ADASYN_all = [targetsLinSVM128ADASYN_all; squeeze(targetLinSVM128ADASYN)];
    predsLinSVM128ADASYN_all = [predsLinSVM128ADASYN_all; squeeze(predsLinSVM128ADASYN)];
    
    [~,scoresLinSVM128ADASYN] = resubPredict(fitPosterior(classificationLinearSVM128ADASYN));
    [xLinSVM128ADASYN,yLinSVM128ADASYN,~,aucLinSVM128ADASYN] = perfcurve(trainingData(train,end),scoresLinSVM128ADASYN(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM128ADASYN.mat','classificationLinearSVM128ADASYN','-v7.3'); % majority voting
    %save('classificationLinearSVM128ADASYN_30.mat','classificationLinearSVM128ADASYN','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM128ADASYN_20.mat','classificationLinearSVM128ADASYN','-v7.3'); % Threshold 20%
    save('classificationLinearSVM128ADASYN_10.mat','classificationLinearSVM128ADASYN','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM128ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM128ADASYN,~] = predict(classificationQuadSVM128ADASYN,trainingData(test,1:end-1));
    targetQuadSVM128ADASYN = trainingData(test,end);
    targetsQuadSVM128ADASYN_all = [targetsQuadSVM128ADASYN_all; squeeze(targetQuadSVM128ADASYN)];
    predsQuadSVM128ADASYN_all = [predsQuadSVM128ADASYN_all; squeeze(predsQuadSVM128ADASYN)];
    
    [~,scoresQuadSVM128ADASYN] = resubPredict(fitPosterior(classificationQuadSVM128ADASYN));
    [xQuadSVM128ADASYN,yQuadSVM128ADASYN,~,aucQuadSVM128ADASYN] = perfcurve(trainingData(train,end),scoresQuadSVM128ADASYN(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM128ADASYN.mat','classificationQuadSVM128ADASYN','-v7.3');
    %save('classificationQuadSVM128ADASYN_30.mat','classificationQuadSVM128ADASYN','-v7.3');
    %save('classificationQuadSVM128ADASYN_20.mat','classificationQuadSVM128ADASYN','-v7.3');
    save('classificationQuadSVM128ADASYN_10.mat','classificationQuadSVM128ADASYN','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM128ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM128ADASYN,~] = predict(classificationCubicSVM128ADASYN,trainingData(test,1:end-1));
    targetCubicSVM128ADASYN = trainingData(test,end);
    targetsCubicSVM128ADASYN_all = [targetsCubicSVM128ADASYN_all; squeeze(targetCubicSVM128ADASYN)];
    predsCubicSVM128ADASYN_all = [predsCubicSVM128ADASYN_all; squeeze(predsCubicSVM128ADASYN)];
    
    [~,scoresCubicSVM128ADASYN] = resubPredict(fitPosterior(classificationCubicSVM128ADASYN));
    [xCubicSVM128ADASYN,yCubicSVM128ADASYN,~,aucCubicSVM128ADASYN] = perfcurve(trainingData(train,end),scoresCubicSVM128ADASYN(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM128ADASYN.mat','classificationCubicSVM128ADASYN','-v7.3');
    %save('classificationCubicSVM128ADASYN_30.mat','classificationCubicSVM128ADASYN','-v7.3');
    %save('classificationCubicSVM128ADASYN_20.mat','classificationCubicSVM128ADASYN','-v7.3');
    save('classificationCubicSVM128ADASYN_10.mat','classificationCubicSVM128ADASYN','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM128ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM128ADASYN,~] = predict(classificationFineSVM128ADASYN,trainingData(test,1:end-1));
    targetsFineSVM128ADASYN = trainingData(test,end);
    targetsFineSVM128ADASYN_all = [targetsFineSVM128ADASYN_all; squeeze(targetsFineSVM128ADASYN)];
    predsFineSVM128ADASYN_all = [predsFineSVM128ADASYN_all; squeeze(predsFineSVM128ADASYN)];
    
    [~,scoresFineSVM128ADASYN] = resubPredict(fitPosterior(classificationFineSVM128ADASYN));
    [xFineSVM128ADASYN,yFineSVM128ADASYN,~,aucFineSVM128ADASYN] = perfcurve(trainingData(train,end),scoresFineSVM128ADASYN(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM128ADASYN.mat','classificationFineSVM128ADASYN','-v7.3');
    %save('classificationFineSVM128ADASYN_30.mat','classificationFineSVM128ADASYN','-v7.3');
    %save('classificationFineSVM128ADASYN_20.mat','classificationFineSVM128ADASYN','-v7.3');
    save('classificationFineSVM128ADASYN_10.mat','classificationFineSVM128ADASYN','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM128ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM128ADASYN,~] = predict(classificationMediumSVM128ADASYN,trainingData(test,1:end-1));
    targetsMediumSVM128ADASYN = trainingData(test,end);
    targetsMediumSVM128ADASYN_all = [targetsMediumSVM128ADASYN_all; squeeze(targetsMediumSVM128ADASYN)];
    predsMediumSVM128ADASYN_all = [predsMediumSVM128ADASYN_all; squeeze(predsMediumSVM128ADASYN)];
    
    [~,scoresMediumSVM128ADASYN] = resubPredict(fitPosterior(classificationMediumSVM128ADASYN));
    [xMediumSVM128ADASYN,yMediumSVM128ADASYN,~,aucMediumSVM128ADASYN] = perfcurve(trainingData(train,end),scoresMediumSVM128ADASYN(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM128ADASYN.mat','classificationMediumSVM128ADASYN','-v7.3');
    %save('classificationMediumSVM128ADASYN_30.mat','classificationMediumSVM128ADASYN','-v7.3');
    %save('classificationMediumSVM128ADASYN_20.mat','classificationMediumSVM128ADASYN','-v7.3');
    save('classificationMediumSVM128ADASYN_10.mat','classificationMediumSVM128ADASYN','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM128ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM128ADASYN,~] = predict(classificationCoarseSVM128ADASYN,trainingData(test,1:end-1));
    targetsCoarseSVM128ADASYN = trainingData(test,end);
    targetsCoarseSVM128ADASYN_all = [targetsCoarseSVM128ADASYN_all; squeeze(targetsCoarseSVM128ADASYN)];
    predsCoarseSVM128ADASYN_all = [predsCoarseSVM128ADASYN_all; squeeze(predsCoarseSVM128ADASYN)];
    
    [~,scoresCoarseSVM128ADASYN] = resubPredict(fitPosterior(classificationCoarseSVM128ADASYN));
    [xCoarseSVM128ADASYN,yCoarseSVM128ADASYN,~,aucCoarseSVM128ADASYN] = perfcurve(trainingData(train,end),scoresCoarseSVM128ADASYN(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM128ADASYN.mat','classificationCoarseSVM128ADASYN','-v7.3');
    %save('classificationCoarseSVM128ADASYN_30.mat','classificationCoarseSVM128ADASYN','-v7.3');
    %save('classificationCoarseSVM128ADASYN_20.mat','classificationCoarseSVM128ADASYN','-v7.3');
    save('classificationCoarseSVM128ADASYN_10.mat','classificationCoarseSVM128ADASYN','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN128ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN128ADASYN,~] = predict(classification3KNN128ADASYN,trainingData(test,1:end-1));
    targets3KNN128ADASYN = trainingData(test,end);
    targets3KNN128ADASYN_all = [targets3KNN128ADASYN_all; squeeze(targets3KNN128ADASYN)];
    preds3KNN128ADASYN_all = [preds3KNN128ADASYN_all; squeeze(preds3KNN128ADASYN)];
    
    [~,scores3KNN128ADASYN] = resubPredict((classification3KNN128ADASYN));
    [x3KNN128ADASYN,y3KNN128ADASYN,~,auc3KNN128ADASYN] = perfcurve(trainingData(train,end),scores3KNN128ADASYN(:,2),1);
    t7 = toc;
    
    %save('classification3KNN128ADASYN.mat','classification3KNN128ADASYN','-v7.3');
    %save('classification3KNN128ADASYN_30.mat','classification3KNN128ADASYN','-v7.3');
    %save('classification3KNN128ADASYN_20.mat','classification3KNN128ADASYN','-v7.3');
    save('classification3KNN128ADASYN_10.mat','classification3KNN128ADASYN','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN128ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN128ADASYN,~] = predict(classification5KNN128ADASYN,trainingData(test,1:end-1));
    targets5KNN128ADASYN = trainingData(test,end);
    targets5KNN128ADASYN_all = [targets5KNN128ADASYN_all; squeeze(targets5KNN128ADASYN)];
    preds5KNN128ADASYN_all = [preds5KNN128ADASYN_all; squeeze(preds5KNN128ADASYN)];
    
    [~,scores5KNN128ADASYN] = resubPredict((classification5KNN128ADASYN));
    [x5KNN128ADASYN,y5KNN128ADASYN,~,auc5KNN128ADASYN] = perfcurve(trainingData(train,end),scores5KNN128ADASYN(:,2),1);
    t8 = toc;

    %save('classification5KNN128ADASYN.mat','classification5KNN128ADASYN','-v7.3');
    %save('classification5KNN128ADASYN_30.mat','classification5KNN128ADASYN','-v7.3');
    %save('classification5KNN128ADASYN_20.mat','classification5KNN128ADASYN','-v7.3');
    save('classification5KNN128ADASYN_10.mat','classification5KNN128ADASYN','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN128ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN128ADASYN,~] = predict(classification7KNN128ADASYN,trainingData(test,1:end-1));
    targets7KNN128ADASYN = trainingData(test,end);
    targets7KNN128ADASYN_all = [targets7KNN128ADASYN_all; squeeze(targets7KNN128ADASYN)];
    preds7KNN128ADASYN_all = [preds7KNN128ADASYN_all; squeeze(preds7KNN128ADASYN)];
    
    [~,scores7KNN128ADASYN] = resubPredict((classification7KNN128ADASYN));
    [x7KNN128ADASYN,y7KNN128ADASYN,~,auc7KNN128ADASYN] = perfcurve(trainingData(train,end),scores7KNN128ADASYN(:,2),1);
    t9 = toc;
    
    %save('classification7KNN128ADASYN.mat','classification7KNN128ADASYN','-v7.3');
    %save('classification7KNN128ADASYN_30.mat','classification7KNN128ADASYN','-v7.3');
    %save('classification7KNN128ADASYN_20.mat','classification7KNN128ADASYN','-v7.3');
    save('classification7KNN128ADASYN_10.mat','classification7KNN128ADASYN','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN128ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN128ADASYN,~] = predict(classification9KNN128ADASYN,trainingData(test,1:end-1));
    targets9KNN128ADASYN = trainingData(test,end);
    targets9KNN128ADASYN_all = [targets9KNN128ADASYN_all; squeeze(targets9KNN128ADASYN)];
    preds9KNN128ADASYN_all = [preds9KNN128ADASYN_all; squeeze(preds9KNN128ADASYN)];
    
    [~,scores9KNN128ADASYN] = resubPredict((classification9KNN128ADASYN));
    [x9KNN128ADASYN,y9KNN128ADASYN,~,auc9KNN128ADASYN] = perfcurve(trainingData(train,end),scores9KNN128ADASYN(:,2),1);
    t10 = toc;
    
    %save('classification9KNN128ADASYN.mat','classification9KNN128ADASYN','-v7.3');
    %save('classification9KNN128ADASYN_30.mat','classification9KNN128ADASYN','-v7.3');
    %save('classification9KNN128ADASYN_20.mat','classification9KNN128ADASYN','-v7.3');
    save('classification9KNN128ADASYN_10.mat','classification9KNN128ADASYN','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree128ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree128ADASYN,~] = predict(classification5Tree128ADASYN,trainingData(test,1:end-1));
    targets5Tree128ADASYN = trainingData(test,end);
    targets5Tree128ADASYN_all = [targets5Tree128ADASYN_all; squeeze(targets5Tree128ADASYN)];
    preds5Tree128ADASYN_all = [preds5Tree128ADASYN_all; squeeze(preds5Tree128ADASYN)];
    
    [~,scores5Tree128ADASYN] = resubPredict((classification5Tree128ADASYN));
    [x5Tree128ADASYN,y5Tree128ADASYN,~,auc5Tree128ADASYN] = perfcurve(trainingData(train,end),scores5Tree128ADASYN(:,2),1);
    t11 = toc;
    
    %save('classification5Tree128ADASYN.mat','classification5Tree128ADASYN','-v7.3');
    %save('classification5Tree128ADASYN_30.mat','classification5Tree128ADASYN','-v7.3');
    %save('classification5Tree128ADASYN_20.mat','classification5Tree128ADASYN','-v7.3');
    save('classification5Tree128ADASYN_10.mat','classification5Tree128ADASYN','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree128ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree128ADASYN,~] = predict(classification10Tree128ADASYN,trainingData(test,1:end-1));
    targets10Tree128ADASYN = trainingData(test,end);
    targets10Tree128ADASYN_all = [targets10Tree128ADASYN_all; squeeze(targets10Tree128ADASYN)];
    preds10Tree128ADASYN_all = [preds10Tree128ADASYN_all; squeeze(preds10Tree128ADASYN)];
    
    [~,scores10Tree128ADASYN] = resubPredict((classification10Tree128ADASYN));
    [x10Tree128ADASYN,y10Tree128ADASYN,~,auc10Tree128ADASYN] = perfcurve(trainingData(train,end),scores10Tree128ADASYN(:,2),1);
    t12 = toc;
    
    %save('classification10Tree128ADASYN.mat','classification10Tree128ADASYN','-v7.3');
    %save('classification10Tree128ADASYN_30.mat','classification10Tree128ADASYN','-v7.3');
    %save('classification10Tree128ADASYN_20.mat','classification10Tree128ADASYN','-v7.3');
    save('classification10Tree128ADASYN_10.mat','classification10Tree128ADASYN','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree128ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree128ADASYN,~] = predict(classification20Tree128ADASYN,trainingData(test,1:end-1));
    targets20Tree128ADASYN = trainingData(test,end);
    targets20Tree128ADASYN_all = [targets20Tree128ADASYN_all; squeeze(targets20Tree128ADASYN)];
    preds20Tree128ADASYN_all = [preds20Tree128ADASYN_all; squeeze(preds20Tree128ADASYN)];
    [~,scores20Tree128ADASYN] = resubPredict((classification20Tree128ADASYN));
    
    [x20Tree128ADASYN,y20Tree128ADASYN,~,auc20Tree128ADASYN] = perfcurve(trainingData(train,end),scores20Tree128ADASYN(:,2),1);
    t13 = toc;
    
    %save('classification20Tree128ADASYN.mat','classification20Tree128ADASYN','-v7.3');
    %save('classification20Tree128ADASYN_30.mat','classification20Tree128ADASYN','-v7.3');
    %save('classification20Tree128ADASYN_20.mat','classification20Tree128ADASYN','-v7.3');
    save('classification20Tree128ADASYN_10.mat','classification20Tree128ADASYN','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree128ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree128ADASYN,~] = predict(classification30Tree128ADASYN,trainingData(test,1:end-1));
    targets30Tree128ADASYN = trainingData(test,end);
    targets30Tree128ADASYN_all = [targets30Tree128ADASYN_all; squeeze(targets30Tree128ADASYN)];
    preds30Tree128ADASYN_all = [preds30Tree128ADASYN_all; squeeze(preds30Tree128ADASYN)];
    
    [~,scores30Tree128ADASYN] = resubPredict((classification30Tree128ADASYN));
    [x30Tree128ADASYN,y30Tree128ADASYN,~,auc30Tree128ADASYN] = perfcurve(trainingData(train,end),scores30Tree128ADASYN(:,2),1);
    t14 = toc;
    
    %save('classification30Tree128ADASYN.mat','classification30Tree128ADASYN','-v7.3');
    %save('classification30Tree128ADASYN_30.mat','classification30Tree128ADASYN','-v7.3');
    %save('classification30Tree128ADASYN_20.mat','classification30Tree128ADASYN','-v7.3');
    save('classification30Tree128ADASYN_10.mat','classification30Tree128ADASYN','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree128ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree128ADASYN,~] = predict(classification40Tree128ADASYN,trainingData(test,1:end-1));
    targets40Tree128ADASYN = trainingData(test,end);
    targets40Tree128ADASYN_all = [targets40Tree128ADASYN_all; squeeze(targets40Tree128ADASYN)];
    preds40Tree128ADASYN_all = [preds40Tree128ADASYN_all; squeeze(preds40Tree128ADASYN)];
    
    [~,scores40Tree128ADASYN] = resubPredict((classification40Tree128ADASYN));
    [x40Tree128ADASYN,y40Tree128ADASYN,~,auc40Tree128ADASYN] = perfcurve(trainingData(train,end),scores40Tree128ADASYN(:,2),1);
    t15 = toc;
    
    %save('classification40Tree128ADASYN.mat','classification40Tree128ADASYN','-v7.3');
    %save('classification40Tree128ADASYN_30.mat','classification40Tree128ADASYN','-v7.3');
    %save('classification40Tree128ADASYN_20.mat','classification40Tree128ADASYN','-v7.3');
    save('classification40Tree128ADASYN_10.mat','classification40Tree128ADASYN','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree128ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree128ADASYN,~] = predict(classification50Tree128ADASYN,trainingData(test,1:end-1));
    targets50Tree128ADASYN = trainingData(test,end);
    targets50Tree128ADASYN_all = [targets50Tree128ADASYN_all; squeeze(targets50Tree128ADASYN)];
    preds50Tree128ADASYN_all = [preds50Tree128ADASYN_all; squeeze(preds50Tree128ADASYN)];
    
    [~,scores50Tree128ADASYN] = resubPredict((classification50Tree128ADASYN));
    [x50Tree128ADASYN,y50Tree128ADASYN,~,auc50Tree128ADASYN] = perfcurve(trainingData(train,end),scores50Tree128ADASYN(:,2),1);
    t16 = toc;
    
    %save('classification50Tree128ADASYN.mat','classification50Tree128ADASYN','-v7.3');
    %save('classification50Tree128ADASYN_30.mat','classification50Tree128ADASYN','-v7.3');
    %save('classification50Tree128ADASYN_20.mat','classification50Tree128ADASYN','-v7.3');
    save('classification50Tree128ADASYN_10.mat','classification50Tree128ADASYN','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM128ADASYN_all,predsLinSVM128ADASYN_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM128ADASYN_all,predsQuadSVM128ADASYN_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM128ADASYN_all,predsCubicSVM128ADASYN_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM128ADASYN_all,predsFineSVM128ADASYN_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM128ADASYN_all,predsMediumSVM128ADASYN_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM128ADASYN_all,predsCoarseSVM128ADASYN_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN128ADASYN_all,preds3KNN128ADASYN_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN128ADASYN_all,preds5KNN128ADASYN_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN128ADASYN_all,preds7KNN128ADASYN_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN128ADASYN_all,preds9KNN128ADASYN_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree128ADASYN_all,preds5Tree128ADASYN_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree128ADASYN_all,preds10Tree128ADASYN_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree128ADASYN_all,preds20Tree128ADASYN_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree128ADASYN_all,preds30Tree128ADASYN_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree128ADASYN_all,preds40Tree128ADASYN_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree128ADASYN_all,preds50Tree128ADASYN_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM128ADASYN,yLinSVM128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM128ADASYN,yQuadSVM128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM128ADASYN,yCubicSVM128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM128ADASYN,yFineSVM128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM128ADASYN,yMediumSVM128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM128ADASYN,yCoarseSVM128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN128ADASYN,y3KNN128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN128ADASYN,y5KNN128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN128ADASYN,y7KNN128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN128ADASYN,y9KNN128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree128ADASYN,y5Tree128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree128ADASYN,y10Tree128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree128ADASYN,y20Tree128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree128ADASYN,y30Tree128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree128ADASYN,y40Tree128ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree128ADASYN,y50Tree128ADASYN,'LineWidth',2)
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
AUC = [aucLinSVM128ADASYN;aucQuadSVM128ADASYN;aucCubicSVM128ADASYN;aucFineSVM128ADASYN;aucMediumSVM128ADASYN;...
    aucCoarseSVM128ADASYN;auc3KNN128ADASYN;auc5KNN128ADASYN;auc7KNN128ADASYN;auc9KNN128ADASYN;auc5Tree128ADASYN;...
    auc10Tree128ADASYN;auc20Tree128ADASYN;auc30Tree128ADASYN;auc40Tree128ADASYN;auc50Tree128ADASYN];

T = table(AUC,Time,Time_total);

