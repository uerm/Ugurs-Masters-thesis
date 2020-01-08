function T = trainClassifier20ADASYN(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM20ADASYN_all = [];
predsLinSVM20ADASYN_all = [];

targetsQuadSVM20ADASYN_all = [];
predsQuadSVM20ADASYN_all = [];

targetsCubicSVM20ADASYN_all = [];
predsCubicSVM20ADASYN_all = [];

targetsFineSVM20ADASYN_all = [];
predsFineSVM20ADASYN_all = [];

targetsMediumSVM20ADASYN_all = [];
predsMediumSVM20ADASYN_all = [];

targetsCoarseSVM20ADASYN_all = [];
predsCoarseSVM20ADASYN_all = [];

targets3KNN20ADASYN_all = [];
preds3KNN20ADASYN_all = [];

targets5KNN20ADASYN_all = [];
preds5KNN20ADASYN_all = [];

targets7KNN20ADASYN_all = [];
preds7KNN20ADASYN_all = [];

targets9KNN20ADASYN_all = [];
preds9KNN20ADASYN_all = [];

targets5Tree20ADASYN_all = [];
preds5Tree20ADASYN_all = [];

targets10Tree20ADASYN_all = [];
preds10Tree20ADASYN_all = [];

targets20Tree20ADASYN_all = [];
preds20Tree20ADASYN_all = [];

targets30Tree20ADASYN_all = [];
preds30Tree20ADASYN_all = [];

targets40Tree20ADASYN_all = [];
preds40Tree20ADASYN_all = [];

targets50Tree20ADASYN_all = [];
preds50Tree20ADASYN_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM20ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM20ADASYN,~] = predict(classificationLinearSVM20ADASYN,trainingData(test,1:end-1));
    targetLinSVM20ADASYN = trainingData(test,end);
    targetsLinSVM20ADASYN_all = [targetsLinSVM20ADASYN_all; squeeze(targetLinSVM20ADASYN)];
    predsLinSVM20ADASYN_all = [predsLinSVM20ADASYN_all; squeeze(predsLinSVM20ADASYN)];
    
    [~,scoresLinSVM20ADASYN] = resubPredict(fitPosterior(classificationLinearSVM20ADASYN));
    [xLinSVM20ADASYN,yLinSVM20ADASYN,~,aucLinSVM20ADASYN] = perfcurve(trainingData(train,end),scoresLinSVM20ADASYN(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM20ADASYN.mat','classificationLinearSVM20ADASYN','-v7.3'); % majority voting
    %save('classificationLinearSVM20ADASYN_30.mat','classificationLinearSVM20ADASYN','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM20ADASYN_20.mat','classificationLinearSVM20ADASYN','-v7.3'); % Threshold 20%
    save('classificationLinearSVM20ADASYN_10.mat','classificationLinearSVM20ADASYN','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM20ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM20ADASYN,~] = predict(classificationQuadSVM20ADASYN,trainingData(test,1:end-1));
    targetQuadSVM20ADASYN = trainingData(test,end);
    targetsQuadSVM20ADASYN_all = [targetsQuadSVM20ADASYN_all; squeeze(targetQuadSVM20ADASYN)];
    predsQuadSVM20ADASYN_all = [predsQuadSVM20ADASYN_all; squeeze(predsQuadSVM20ADASYN)];
    
    [~,scoresQuadSVM20ADASYN] = resubPredict(fitPosterior(classificationQuadSVM20ADASYN));
    [xQuadSVM20ADASYN,yQuadSVM20ADASYN,~,aucQuadSVM20ADASYN] = perfcurve(trainingData(train,end),scoresQuadSVM20ADASYN(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM20ADASYN.mat','classificationQuadSVM20ADASYN','-v7.3');
    %save('classificationQuadSVM20ADASYN_30.mat','classificationQuadSVM20ADASYN','-v7.3');
    %save('classificationQuadSVM20ADASYN_20.mat','classificationQuadSVM20ADASYN','-v7.3');
    save('classificationQuadSVM20ADASYN_10.mat','classificationQuadSVM20ADASYN','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM20ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM20ADASYN,~] = predict(classificationCubicSVM20ADASYN,trainingData(test,1:end-1));
    targetCubicSVM20ADASYN = trainingData(test,end);
    targetsCubicSVM20ADASYN_all = [targetsCubicSVM20ADASYN_all; squeeze(targetCubicSVM20ADASYN)];
    predsCubicSVM20ADASYN_all = [predsCubicSVM20ADASYN_all; squeeze(predsCubicSVM20ADASYN)];
    
    [~,scoresCubicSVM20ADASYN] = resubPredict(fitPosterior(classificationCubicSVM20ADASYN));
    [xCubicSVM20ADASYN,yCubicSVM20ADASYN,~,aucCubicSVM20ADASYN] = perfcurve(trainingData(train,end),scoresCubicSVM20ADASYN(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM20ADASYN.mat','classificationCubicSVM20ADASYN','-v7.3');
    %save('classificationCubicSVM20ADASYN_30.mat','classificationCubicSVM20ADASYN','-v7.3');
    %save('classificationCubicSVM20ADASYN_20.mat','classificationCubicSVM20ADASYN','-v7.3');
    save('classificationCubicSVM20ADASYN_10.mat','classificationCubicSVM20ADASYN','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM20ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM20ADASYN,~] = predict(classificationFineSVM20ADASYN,trainingData(test,1:end-1));
    targetsFineSVM20ADASYN = trainingData(test,end);
    targetsFineSVM20ADASYN_all = [targetsFineSVM20ADASYN_all; squeeze(targetsFineSVM20ADASYN)];
    predsFineSVM20ADASYN_all = [predsFineSVM20ADASYN_all; squeeze(predsFineSVM20ADASYN)];
    
    [~,scoresFineSVM20ADASYN] = resubPredict(fitPosterior(classificationFineSVM20ADASYN));
    [xFineSVM20ADASYN,yFineSVM20ADASYN,~,aucFineSVM20ADASYN] = perfcurve(trainingData(train,end),scoresFineSVM20ADASYN(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM20ADASYN.mat','classificationFineSVM20ADASYN','-v7.3');
    %save('classificationFineSVM20ADASYN_30.mat','classificationFineSVM20ADASYN','-v7.3');
    %save('classificationFineSVM20ADASYN_20.mat','classificationFineSVM20ADASYN','-v7.3');
    save('classificationFineSVM20ADASYN_10.mat','classificationFineSVM20ADASYN','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM20ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM20ADASYN,~] = predict(classificationMediumSVM20ADASYN,trainingData(test,1:end-1));
    targetsMediumSVM20ADASYN = trainingData(test,end);
    targetsMediumSVM20ADASYN_all = [targetsMediumSVM20ADASYN_all; squeeze(targetsMediumSVM20ADASYN)];
    predsMediumSVM20ADASYN_all = [predsMediumSVM20ADASYN_all; squeeze(predsMediumSVM20ADASYN)];
    
    [~,scoresMediumSVM20ADASYN] = resubPredict(fitPosterior(classificationMediumSVM20ADASYN));
    [xMediumSVM20ADASYN,yMediumSVM20ADASYN,~,aucMediumSVM20ADASYN] = perfcurve(trainingData(train,end),scoresMediumSVM20ADASYN(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM20ADASYN.mat','classificationMediumSVM20ADASYN','-v7.3');
    %save('classificationMediumSVM20ADASYN_30.mat','classificationMediumSVM20ADASYN','-v7.3');
    %save('classificationMediumSVM20ADASYN_20.mat','classificationMediumSVM20ADASYN','-v7.3');
    save('classificationMediumSVM20ADASYN_10.mat','classificationMediumSVM20ADASYN','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM20ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM20ADASYN,~] = predict(classificationCoarseSVM20ADASYN,trainingData(test,1:end-1));
    targetsCoarseSVM20ADASYN = trainingData(test,end);
    targetsCoarseSVM20ADASYN_all = [targetsCoarseSVM20ADASYN_all; squeeze(targetsCoarseSVM20ADASYN)];
    predsCoarseSVM20ADASYN_all = [predsCoarseSVM20ADASYN_all; squeeze(predsCoarseSVM20ADASYN)];
    
    [~,scoresCoarseSVM20ADASYN] = resubPredict(fitPosterior(classificationCoarseSVM20ADASYN));
    [xCoarseSVM20ADASYN,yCoarseSVM20ADASYN,~,aucCoarseSVM20ADASYN] = perfcurve(trainingData(train,end),scoresCoarseSVM20ADASYN(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM20ADASYN.mat','classificationCoarseSVM20ADASYN','-v7.3');
    %save('classificationCoarseSVM20ADASYN_30.mat','classificationCoarseSVM20ADASYN','-v7.3');
    %save('classificationCoarseSVM20ADASYN_20.mat','classificationCoarseSVM20ADASYN','-v7.3');
    save('classificationCoarseSVM20ADASYN_10.mat','classificationCoarseSVM20ADASYN','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN20ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN20ADASYN,~] = predict(classification3KNN20ADASYN,trainingData(test,1:end-1));
    targets3KNN20ADASYN = trainingData(test,end);
    targets3KNN20ADASYN_all = [targets3KNN20ADASYN_all; squeeze(targets3KNN20ADASYN)];
    preds3KNN20ADASYN_all = [preds3KNN20ADASYN_all; squeeze(preds3KNN20ADASYN)];
    
    [~,scores3KNN20ADASYN] = resubPredict((classification3KNN20ADASYN));
    [x3KNN20ADASYN,y3KNN20ADASYN,~,auc3KNN20ADASYN] = perfcurve(trainingData(train,end),scores3KNN20ADASYN(:,2),1);
    t7 = toc;
    
    %save('classification3KNN20ADASYN.mat','classification3KNN20ADASYN','-v7.3');
    %save('classification3KNN20ADASYN_30.mat','classification3KNN20ADASYN','-v7.3');
    %save('classification3KNN20ADASYN_20.mat','classification3KNN20ADASYN','-v7.3');
    save('classification3KNN20ADASYN_10.mat','classification3KNN20ADASYN','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN20ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN20ADASYN,~] = predict(classification5KNN20ADASYN,trainingData(test,1:end-1));
    targets5KNN20ADASYN = trainingData(test,end);
    targets5KNN20ADASYN_all = [targets5KNN20ADASYN_all; squeeze(targets5KNN20ADASYN)];
    preds5KNN20ADASYN_all = [preds5KNN20ADASYN_all; squeeze(preds5KNN20ADASYN)];
    
    [~,scores5KNN20ADASYN] = resubPredict((classification5KNN20ADASYN));
    [x5KNN20ADASYN,y5KNN20ADASYN,~,auc5KNN20ADASYN] = perfcurve(trainingData(train,end),scores5KNN20ADASYN(:,2),1);
    t8 = toc;

    %save('classification5KNN20ADASYN.mat','classification5KNN20ADASYN','-v7.3');
    %save('classification5KNN20ADASYN_30.mat','classification5KNN20ADASYN','-v7.3');
    %save('classification5KNN20ADASYN_20.mat','classification5KNN20ADASYN','-v7.3');
    save('classification5KNN20ADASYN_10.mat','classification5KNN20ADASYN','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN20ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN20ADASYN,~] = predict(classification7KNN20ADASYN,trainingData(test,1:end-1));
    targets7KNN20ADASYN = trainingData(test,end);
    targets7KNN20ADASYN_all = [targets7KNN20ADASYN_all; squeeze(targets7KNN20ADASYN)];
    preds7KNN20ADASYN_all = [preds7KNN20ADASYN_all; squeeze(preds7KNN20ADASYN)];
    
    [~,scores7KNN20ADASYN] = resubPredict((classification7KNN20ADASYN));
    [x7KNN20ADASYN,y7KNN20ADASYN,~,auc7KNN20ADASYN] = perfcurve(trainingData(train,end),scores7KNN20ADASYN(:,2),1);
    t9 = toc;
    
    %save('classification7KNN20ADASYN.mat','classification7KNN20ADASYN','-v7.3');
    %save('classification7KNN20ADASYN_30.mat','classification7KNN20ADASYN','-v7.3');
    %save('classification7KNN20ADASYN_20.mat','classification7KNN20ADASYN','-v7.3');
    save('classification7KNN20ADASYN_10.mat','classification7KNN20ADASYN','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN20ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN20ADASYN,~] = predict(classification9KNN20ADASYN,trainingData(test,1:end-1));
    targets9KNN20ADASYN = trainingData(test,end);
    targets9KNN20ADASYN_all = [targets9KNN20ADASYN_all; squeeze(targets9KNN20ADASYN)];
    preds9KNN20ADASYN_all = [preds9KNN20ADASYN_all; squeeze(preds9KNN20ADASYN)];
    
    [~,scores9KNN20ADASYN] = resubPredict((classification9KNN20ADASYN));
    [x9KNN20ADASYN,y9KNN20ADASYN,~,auc9KNN20ADASYN] = perfcurve(trainingData(train,end),scores9KNN20ADASYN(:,2),1);
    t10 = toc;
    
    %save('classification9KNN20ADASYN.mat','classification9KNN20ADASYN','-v7.3');
    %save('classification9KNN20ADASYN_30.mat','classification9KNN20ADASYN','-v7.3');
    %save('classification9KNN20ADASYN_20.mat','classification9KNN20ADASYN','-v7.3');
    save('classification9KNN20ADASYN_10.mat','classification9KNN20ADASYN','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree20ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree20ADASYN,~] = predict(classification5Tree20ADASYN,trainingData(test,1:end-1));
    targets5Tree20ADASYN = trainingData(test,end);
    targets5Tree20ADASYN_all = [targets5Tree20ADASYN_all; squeeze(targets5Tree20ADASYN)];
    preds5Tree20ADASYN_all = [preds5Tree20ADASYN_all; squeeze(preds5Tree20ADASYN)];
    
    [~,scores5Tree20ADASYN] = resubPredict((classification5Tree20ADASYN));
    [x5Tree20ADASYN,y5Tree20ADASYN,~,auc5Tree20ADASYN] = perfcurve(trainingData(train,end),scores5Tree20ADASYN(:,2),1);
    t11 = toc;
    
    %save('classification5Tree20ADASYN.mat','classification5Tree20ADASYN','-v7.3');
    %save('classification5Tree20ADASYN_30.mat','classification5Tree20ADASYN','-v7.3');
    %save('classification5Tree20ADASYN_20.mat','classification5Tree20ADASYN','-v7.3');
    save('classification5Tree20ADASYN_10.mat','classification5Tree20ADASYN','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree20ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree20ADASYN,~] = predict(classification10Tree20ADASYN,trainingData(test,1:end-1));
    targets10Tree20ADASYN = trainingData(test,end);
    targets10Tree20ADASYN_all = [targets10Tree20ADASYN_all; squeeze(targets10Tree20ADASYN)];
    preds10Tree20ADASYN_all = [preds10Tree20ADASYN_all; squeeze(preds10Tree20ADASYN)];
    
    [~,scores10Tree20ADASYN] = resubPredict((classification10Tree20ADASYN));
    [x10Tree20ADASYN,y10Tree20ADASYN,~,auc10Tree20ADASYN] = perfcurve(trainingData(train,end),scores10Tree20ADASYN(:,2),1);
    t12 = toc;
    
    %save('classification10Tree20ADASYN.mat','classification10Tree20ADASYN','-v7.3');
    %save('classification10Tree20ADASYN_30.mat','classification10Tree20ADASYN','-v7.3');
    %save('classification10Tree20ADASYN_20.mat','classification10Tree20ADASYN','-v7.3');
    save('classification10Tree20ADASYN_10.mat','classification10Tree20ADASYN','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree20ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree20ADASYN,~] = predict(classification20Tree20ADASYN,trainingData(test,1:end-1));
    targets20Tree20ADASYN = trainingData(test,end);
    targets20Tree20ADASYN_all = [targets20Tree20ADASYN_all; squeeze(targets20Tree20ADASYN)];
    preds20Tree20ADASYN_all = [preds20Tree20ADASYN_all; squeeze(preds20Tree20ADASYN)];
    [~,scores20Tree20ADASYN] = resubPredict((classification20Tree20ADASYN));
    
    [x20Tree20ADASYN,y20Tree20ADASYN,~,auc20Tree20ADASYN] = perfcurve(trainingData(train,end),scores20Tree20ADASYN(:,2),1);
    t13 = toc;
    
    %save('classification20Tree20ADASYN.mat','classification20Tree20ADASYN','-v7.3');
    %save('classification20Tree20ADASYN_30.mat','classification20Tree20ADASYN','-v7.3');
    %save('classification20Tree20ADASYN_20.mat','classification20Tree20ADASYN','-v7.3');
    save('classification20Tree20ADASYN_10.mat','classification20Tree20ADASYN','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree20ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree20ADASYN,~] = predict(classification30Tree20ADASYN,trainingData(test,1:end-1));
    targets30Tree20ADASYN = trainingData(test,end);
    targets30Tree20ADASYN_all = [targets30Tree20ADASYN_all; squeeze(targets30Tree20ADASYN)];
    preds30Tree20ADASYN_all = [preds30Tree20ADASYN_all; squeeze(preds30Tree20ADASYN)];
    
    [~,scores30Tree20ADASYN] = resubPredict((classification30Tree20ADASYN));
    [x30Tree20ADASYN,y30Tree20ADASYN,~,auc30Tree20ADASYN] = perfcurve(trainingData(train,end),scores30Tree20ADASYN(:,2),1);
    t14 = toc;
    
    %save('classification30Tree20ADASYN.mat','classification30Tree20ADASYN','-v7.3');
    %save('classification30Tree20ADASYN_30.mat','classification30Tree20ADASYN','-v7.3');
    %save('classification30Tree20ADASYN_20.mat','classification30Tree20ADASYN','-v7.3');
    save('classification30Tree20ADASYN_10.mat','classification30Tree20ADASYN','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree20ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree20ADASYN,~] = predict(classification40Tree20ADASYN,trainingData(test,1:end-1));
    targets40Tree20ADASYN = trainingData(test,end);
    targets40Tree20ADASYN_all = [targets40Tree20ADASYN_all; squeeze(targets40Tree20ADASYN)];
    preds40Tree20ADASYN_all = [preds40Tree20ADASYN_all; squeeze(preds40Tree20ADASYN)];
    
    [~,scores40Tree20ADASYN] = resubPredict((classification40Tree20ADASYN));
    [x40Tree20ADASYN,y40Tree20ADASYN,~,auc40Tree20ADASYN] = perfcurve(trainingData(train,end),scores40Tree20ADASYN(:,2),1);
    t15 = toc;
    
    %save('classification40Tree20ADASYN.mat','classification40Tree20ADASYN','-v7.3');
    %save('classification40Tree20ADASYN_30.mat','classification40Tree20ADASYN','-v7.3');
    %save('classification40Tree20ADASYN_20.mat','classification40Tree20ADASYN','-v7.3');
    save('classification40Tree20ADASYN_10.mat','classification40Tree20ADASYN','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree20ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree20ADASYN,~] = predict(classification50Tree20ADASYN,trainingData(test,1:end-1));
    targets50Tree20ADASYN = trainingData(test,end);
    targets50Tree20ADASYN_all = [targets50Tree20ADASYN_all; squeeze(targets50Tree20ADASYN)];
    preds50Tree20ADASYN_all = [preds50Tree20ADASYN_all; squeeze(preds50Tree20ADASYN)];
    
    [~,scores50Tree20ADASYN] = resubPredict((classification50Tree20ADASYN));
    [x50Tree20ADASYN,y50Tree20ADASYN,~,auc50Tree20ADASYN] = perfcurve(trainingData(train,end),scores50Tree20ADASYN(:,2),1);
    t16 = toc;
    
    %save('classification50Tree20ADASYN.mat','classification50Tree20ADASYN','-v7.3');
    %save('classification50Tree20ADASYN_30.mat','classification50Tree20ADASYN','-v7.3');
    %save('classification50Tree20ADASYN_20.mat','classification50Tree20ADASYN','-v7.3');
    save('classification50Tree20ADASYN_10.mat','classification50Tree20ADASYN','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM20ADASYN_all,predsLinSVM20ADASYN_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM20ADASYN_all,predsQuadSVM20ADASYN_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM20ADASYN_all,predsCubicSVM20ADASYN_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM20ADASYN_all,predsFineSVM20ADASYN_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM20ADASYN_all,predsMediumSVM20ADASYN_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM20ADASYN_all,predsCoarseSVM20ADASYN_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN20ADASYN_all,preds3KNN20ADASYN_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN20ADASYN_all,preds5KNN20ADASYN_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN20ADASYN_all,preds7KNN20ADASYN_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN20ADASYN_all,preds9KNN20ADASYN_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree20ADASYN_all,preds5Tree20ADASYN_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree20ADASYN_all,preds10Tree20ADASYN_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree20ADASYN_all,preds20Tree20ADASYN_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree20ADASYN_all,preds30Tree20ADASYN_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree20ADASYN_all,preds40Tree20ADASYN_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree20ADASYN_all,preds50Tree20ADASYN_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM20ADASYN,yLinSVM20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM20ADASYN,yQuadSVM20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM20ADASYN,yCubicSVM20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM20ADASYN,yFineSVM20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM20ADASYN,yMediumSVM20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM20ADASYN,yCoarseSVM20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN20ADASYN,y3KNN20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN20ADASYN,y5KNN20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN20ADASYN,y7KNN20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN20ADASYN,y9KNN20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree20ADASYN,y5Tree20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree20ADASYN,y10Tree20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree20ADASYN,y20Tree20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree20ADASYN,y30Tree20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree20ADASYN,y40Tree20ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree20ADASYN,y50Tree20ADASYN,'LineWidth',2)
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
AUC = [aucLinSVM20ADASYN;aucQuadSVM20ADASYN;aucCubicSVM20ADASYN;aucFineSVM20ADASYN;aucMediumSVM20ADASYN;...
    aucCoarseSVM20ADASYN;auc3KNN20ADASYN;auc5KNN20ADASYN;auc7KNN20ADASYN;auc9KNN20ADASYN;auc5Tree20ADASYN;...
    auc10Tree20ADASYN;auc20Tree20ADASYN;auc30Tree20ADASYN;auc40Tree20ADASYN;auc50Tree20ADASYN];

T = table(AUC,Time,Time_total);

