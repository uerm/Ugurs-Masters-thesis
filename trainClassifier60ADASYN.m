function T = trainClassifier60ADASYN(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM60ADASYN_all = [];
predsLinSVM60ADASYN_all = [];

targetsQuadSVM60ADASYN_all = [];
predsQuadSVM60ADASYN_all = [];

targetsCubicSVM60ADASYN_all = [];
predsCubicSVM60ADASYN_all = [];

targetsFineSVM60ADASYN_all = [];
predsFineSVM60ADASYN_all = [];

targetsMediumSVM60ADASYN_all = [];
predsMediumSVM60ADASYN_all = [];

targetsCoarseSVM60ADASYN_all = [];
predsCoarseSVM60ADASYN_all = [];

targets3KNN60ADASYN_all = [];
preds3KNN60ADASYN_all = [];

targets5KNN60ADASYN_all = [];
preds5KNN60ADASYN_all = [];

targets7KNN60ADASYN_all = [];
preds7KNN60ADASYN_all = [];

targets9KNN60ADASYN_all = [];
preds9KNN60ADASYN_all = [];

targets5Tree60ADASYN_all = [];
preds5Tree60ADASYN_all = [];

targets10Tree60ADASYN_all = [];
preds10Tree60ADASYN_all = [];

targets20Tree60ADASYN_all = [];
preds20Tree60ADASYN_all = [];

targets30Tree60ADASYN_all = [];
preds30Tree60ADASYN_all = [];

targets40Tree60ADASYN_all = [];
preds40Tree60ADASYN_all = [];

targets50Tree60ADASYN_all = [];
preds50Tree60ADASYN_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM60ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM60ADASYN,~] = predict(classificationLinearSVM60ADASYN,trainingData(test,1:end-1));
    targetLinSVM60ADASYN = trainingData(test,end);
    targetsLinSVM60ADASYN_all = [targetsLinSVM60ADASYN_all; squeeze(targetLinSVM60ADASYN)];
    predsLinSVM60ADASYN_all = [predsLinSVM60ADASYN_all; squeeze(predsLinSVM60ADASYN)];
    
    [~,scoresLinSVM60ADASYN] = resubPredict(fitPosterior(classificationLinearSVM60ADASYN));
    [xLinSVM60ADASYN,yLinSVM60ADASYN,~,aucLinSVM60ADASYN] = perfcurve(trainingData(train,end),scoresLinSVM60ADASYN(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM60ADASYN.mat','classificationLinearSVM60ADASYN','-v7.3'); % majority voting
    save('classificationLinearSVM60ADASYN_30.mat','classificationLinearSVM60ADASYN','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM60ADASYN_20.mat','classificationLinearSVM60ADASYN','-v7.3'); % Threshold 20%
    %save('classificationLinearSVM60ADASYN_10.mat','classificationLinearSVM60ADASYN','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM60ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM60ADASYN,~] = predict(classificationQuadSVM60ADASYN,trainingData(test,1:end-1));
    targetQuadSVM60ADASYN = trainingData(test,end);
    targetsQuadSVM60ADASYN_all = [targetsQuadSVM60ADASYN_all; squeeze(targetQuadSVM60ADASYN)];
    predsQuadSVM60ADASYN_all = [predsQuadSVM60ADASYN_all; squeeze(predsQuadSVM60ADASYN)];
    
    [~,scoresQuadSVM60ADASYN] = resubPredict(fitPosterior(classificationQuadSVM60ADASYN));
    [xQuadSVM60ADASYN,yQuadSVM60ADASYN,~,aucQuadSVM60ADASYN] = perfcurve(trainingData(train,end),scoresQuadSVM60ADASYN(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM60ADASYN.mat','classificationQuadSVM60ADASYN','-v7.3');
    save('classificationQuadSVM60ADASYN_30.mat','classificationQuadSVM60ADASYN','-v7.3');
    %save('classificationQuadSVM60ADASYN_20.mat','classificationQuadSVM60ADASYN','-v7.3');
    %save('classificationQuadSVM60ADASYN_10.mat','classificationQuadSVM60ADASYN','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM60ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM60ADASYN,~] = predict(classificationCubicSVM60ADASYN,trainingData(test,1:end-1));
    targetCubicSVM60ADASYN = trainingData(test,end);
    targetsCubicSVM60ADASYN_all = [targetsCubicSVM60ADASYN_all; squeeze(targetCubicSVM60ADASYN)];
    predsCubicSVM60ADASYN_all = [predsCubicSVM60ADASYN_all; squeeze(predsCubicSVM60ADASYN)];
    
    [~,scoresCubicSVM60ADASYN] = resubPredict(fitPosterior(classificationCubicSVM60ADASYN));
    [xCubicSVM60ADASYN,yCubicSVM60ADASYN,~,aucCubicSVM60ADASYN] = perfcurve(trainingData(train,end),scoresCubicSVM60ADASYN(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM60ADASYN.mat','classificationCubicSVM60ADASYN','-v7.3');
    save('classificationCubicSVM60ADASYN_30.mat','classificationCubicSVM60ADASYN','-v7.3');
    %save('classificationCubicSVM60ADASYN_20.mat','classificationCubicSVM60ADASYN','-v7.3');
    %save('classificationCubicSVM60ADASYN_10.mat','classificationCubicSVM60ADASYN','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM60ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM60ADASYN,~] = predict(classificationFineSVM60ADASYN,trainingData(test,1:end-1));
    targetsFineSVM60ADASYN = trainingData(test,end);
    targetsFineSVM60ADASYN_all = [targetsFineSVM60ADASYN_all; squeeze(targetsFineSVM60ADASYN)];
    predsFineSVM60ADASYN_all = [predsFineSVM60ADASYN_all; squeeze(predsFineSVM60ADASYN)];
    
    [~,scoresFineSVM60ADASYN] = resubPredict(fitPosterior(classificationFineSVM60ADASYN));
    [xFineSVM60ADASYN,yFineSVM60ADASYN,~,aucFineSVM60ADASYN] = perfcurve(trainingData(train,end),scoresFineSVM60ADASYN(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM60ADASYN.mat','classificationFineSVM60ADASYN','-v7.3');
    save('classificationFineSVM60ADASYN_30.mat','classificationFineSVM60ADASYN','-v7.3');
    %save('classificationFineSVM60ADASYN_20.mat','classificationFineSVM60ADASYN','-v7.3');
    %save('classificationFineSVM60ADASYN_10.mat','classificationFineSVM60ADASYN','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM60ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM60ADASYN,~] = predict(classificationMediumSVM60ADASYN,trainingData(test,1:end-1));
    targetsMediumSVM60ADASYN = trainingData(test,end);
    targetsMediumSVM60ADASYN_all = [targetsMediumSVM60ADASYN_all; squeeze(targetsMediumSVM60ADASYN)];
    predsMediumSVM60ADASYN_all = [predsMediumSVM60ADASYN_all; squeeze(predsMediumSVM60ADASYN)];
    
    [~,scoresMediumSVM60ADASYN] = resubPredict(fitPosterior(classificationMediumSVM60ADASYN));
    [xMediumSVM60ADASYN,yMediumSVM60ADASYN,~,aucMediumSVM60ADASYN] = perfcurve(trainingData(train,end),scoresMediumSVM60ADASYN(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM60ADASYN.mat','classificationMediumSVM60ADASYN','-v7.3');
    save('classificationMediumSVM60ADASYN_30.mat','classificationMediumSVM60ADASYN','-v7.3');
    %save('classificationMediumSVM60ADASYN_20.mat','classificationMediumSVM60ADASYN','-v7.3');
    %save('classificationMediumSVM60ADASYN_10.mat','classificationMediumSVM60ADASYN','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM60ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM60ADASYN,~] = predict(classificationCoarseSVM60ADASYN,trainingData(test,1:end-1));
    targetsCoarseSVM60ADASYN = trainingData(test,end);
    targetsCoarseSVM60ADASYN_all = [targetsCoarseSVM60ADASYN_all; squeeze(targetsCoarseSVM60ADASYN)];
    predsCoarseSVM60ADASYN_all = [predsCoarseSVM60ADASYN_all; squeeze(predsCoarseSVM60ADASYN)];
    
    [~,scoresCoarseSVM60ADASYN] = resubPredict(fitPosterior(classificationCoarseSVM60ADASYN));
    [xCoarseSVM60ADASYN,yCoarseSVM60ADASYN,~,aucCoarseSVM60ADASYN] = perfcurve(trainingData(train,end),scoresCoarseSVM60ADASYN(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM60ADASYN.mat','classificationCoarseSVM60ADASYN','-v7.3');
    save('classificationCoarseSVM60ADASYN_30.mat','classificationCoarseSVM60ADASYN','-v7.3');
    %save('classificationCoarseSVM60ADASYN_20.mat','classificationCoarseSVM60ADASYN','-v7.3');
    %save('classificationCoarseSVM60ADASYN_10.mat','classificationCoarseSVM60ADASYN','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN60ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN60ADASYN,~] = predict(classification3KNN60ADASYN,trainingData(test,1:end-1));
    targets3KNN60ADASYN = trainingData(test,end);
    targets3KNN60ADASYN_all = [targets3KNN60ADASYN_all; squeeze(targets3KNN60ADASYN)];
    preds3KNN60ADASYN_all = [preds3KNN60ADASYN_all; squeeze(preds3KNN60ADASYN)];
    
    [~,scores3KNN60ADASYN] = resubPredict((classification3KNN60ADASYN));
    [x3KNN60ADASYN,y3KNN60ADASYN,~,auc3KNN60ADASYN] = perfcurve(trainingData(train,end),scores3KNN60ADASYN(:,2),1);
    t7 = toc;
    
    %save('classification3KNN60ADASYN.mat','classification3KNN60ADASYN','-v7.3');
    save('classification3KNN60ADASYN_30.mat','classification3KNN60ADASYN','-v7.3');
    %save('classification3KNN60ADASYN_20.mat','classification3KNN60ADASYN','-v7.3');
    %save('classification3KNN60ADASYN_10.mat','classification3KNN60ADASYN','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN60ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN60ADASYN,~] = predict(classification5KNN60ADASYN,trainingData(test,1:end-1));
    targets5KNN60ADASYN = trainingData(test,end);
    targets5KNN60ADASYN_all = [targets5KNN60ADASYN_all; squeeze(targets5KNN60ADASYN)];
    preds5KNN60ADASYN_all = [preds5KNN60ADASYN_all; squeeze(preds5KNN60ADASYN)];
    
    [~,scores5KNN60ADASYN] = resubPredict((classification5KNN60ADASYN));
    [x5KNN60ADASYN,y5KNN60ADASYN,~,auc5KNN60ADASYN] = perfcurve(trainingData(train,end),scores5KNN60ADASYN(:,2),1);
    t8 = toc;

    %save('classification5KNN60ADASYN.mat','classification5KNN60ADASYN','-v7.3');
    save('classification5KNN60ADASYN_30.mat','classification5KNN60ADASYN','-v7.3');
    %save('classification5KNN60ADASYN_20.mat','classification5KNN60ADASYN','-v7.3');
    %save('classification5KNN60ADASYN_10.mat','classification5KNN60ADASYN','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN60ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN60ADASYN,~] = predict(classification7KNN60ADASYN,trainingData(test,1:end-1));
    targets7KNN60ADASYN = trainingData(test,end);
    targets7KNN60ADASYN_all = [targets7KNN60ADASYN_all; squeeze(targets7KNN60ADASYN)];
    preds7KNN60ADASYN_all = [preds7KNN60ADASYN_all; squeeze(preds7KNN60ADASYN)];
    
    [~,scores7KNN60ADASYN] = resubPredict((classification7KNN60ADASYN));
    [x7KNN60ADASYN,y7KNN60ADASYN,~,auc7KNN60ADASYN] = perfcurve(trainingData(train,end),scores7KNN60ADASYN(:,2),1);
    t9 = toc;
    
    %save('classification7KNN60ADASYN.mat','classification7KNN60ADASYN','-v7.3');
    save('classification7KNN60ADASYN_30.mat','classification7KNN60ADASYN','-v7.3');
    %save('classification7KNN60ADASYN_20.mat','classification7KNN60ADASYN','-v7.3');
    %save('classification7KNN60ADASYN_10.mat','classification7KNN60ADASYN','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN60ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN60ADASYN,~] = predict(classification9KNN60ADASYN,trainingData(test,1:end-1));
    targets9KNN60ADASYN = trainingData(test,end);
    targets9KNN60ADASYN_all = [targets9KNN60ADASYN_all; squeeze(targets9KNN60ADASYN)];
    preds9KNN60ADASYN_all = [preds9KNN60ADASYN_all; squeeze(preds9KNN60ADASYN)];
    
    [~,scores9KNN60ADASYN] = resubPredict((classification9KNN60ADASYN));
    [x9KNN60ADASYN,y9KNN60ADASYN,~,auc9KNN60ADASYN] = perfcurve(trainingData(train,end),scores9KNN60ADASYN(:,2),1);
    t10 = toc;
    
    %save('classification9KNN60ADASYN.mat','classification9KNN60ADASYN','-v7.3');
    save('classification9KNN60ADASYN_30.mat','classification9KNN60ADASYN','-v7.3');
    %save('classification9KNN60ADASYN_20.mat','classification9KNN60ADASYN','-v7.3');
    %save('classification9KNN60ADASYN_10.mat','classification9KNN60ADASYN','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree60ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree60ADASYN,~] = predict(classification5Tree60ADASYN,trainingData(test,1:end-1));
    targets5Tree60ADASYN = trainingData(test,end);
    targets5Tree60ADASYN_all = [targets5Tree60ADASYN_all; squeeze(targets5Tree60ADASYN)];
    preds5Tree60ADASYN_all = [preds5Tree60ADASYN_all; squeeze(preds5Tree60ADASYN)];
    
    [~,scores5Tree60ADASYN] = resubPredict((classification5Tree60ADASYN));
    [x5Tree60ADASYN,y5Tree60ADASYN,~,auc5Tree60ADASYN] = perfcurve(trainingData(train,end),scores5Tree60ADASYN(:,2),1);
    t11 = toc;
    
    %save('classification5Tree60ADASYN.mat','classification5Tree60ADASYN','-v7.3');
    save('classification5Tree60ADASYN_30.mat','classification5Tree60ADASYN','-v7.3');
    %save('classification5Tree60ADASYN_20.mat','classification5Tree60ADASYN','-v7.3');
    %save('classification5Tree60ADASYN_10.mat','classification5Tree60ADASYN','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree60ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree60ADASYN,~] = predict(classification10Tree60ADASYN,trainingData(test,1:end-1));
    targets10Tree60ADASYN = trainingData(test,end);
    targets10Tree60ADASYN_all = [targets10Tree60ADASYN_all; squeeze(targets10Tree60ADASYN)];
    preds10Tree60ADASYN_all = [preds10Tree60ADASYN_all; squeeze(preds10Tree60ADASYN)];
    
    [~,scores10Tree60ADASYN] = resubPredict((classification10Tree60ADASYN));
    [x10Tree60ADASYN,y10Tree60ADASYN,~,auc10Tree60ADASYN] = perfcurve(trainingData(train,end),scores10Tree60ADASYN(:,2),1);
    t12 = toc;
    
    %save('classification10Tree60ADASYN.mat','classification10Tree60ADASYN','-v7.3');
    save('classification10Tree60ADASYN_30.mat','classification10Tree60ADASYN','-v7.3');
    %save('classification10Tree60ADASYN_20.mat','classification10Tree60ADASYN','-v7.3');
    %save('classification10Tree60ADASYN_10.mat','classification10Tree60ADASYN','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree60ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree60ADASYN,~] = predict(classification20Tree60ADASYN,trainingData(test,1:end-1));
    targets20Tree60ADASYN = trainingData(test,end);
    targets20Tree60ADASYN_all = [targets20Tree60ADASYN_all; squeeze(targets20Tree60ADASYN)];
    preds20Tree60ADASYN_all = [preds20Tree60ADASYN_all; squeeze(preds20Tree60ADASYN)];
    [~,scores20Tree60ADASYN] = resubPredict((classification20Tree60ADASYN));
    
    [x20Tree60ADASYN,y20Tree60ADASYN,~,auc20Tree60ADASYN] = perfcurve(trainingData(train,end),scores20Tree60ADASYN(:,2),1);
    t13 = toc;
    
    %save('classification20Tree60ADASYN.mat','classification20Tree60ADASYN','-v7.3');
    save('classification20Tree60ADASYN_30.mat','classification20Tree60ADASYN','-v7.3');
    %save('classification20Tree60ADASYN_20.mat','classification20Tree60ADASYN','-v7.3');
    %save('classification20Tree60ADASYN_10.mat','classification20Tree60ADASYN','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree60ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree60ADASYN,~] = predict(classification30Tree60ADASYN,trainingData(test,1:end-1));
    targets30Tree60ADASYN = trainingData(test,end);
    targets30Tree60ADASYN_all = [targets30Tree60ADASYN_all; squeeze(targets30Tree60ADASYN)];
    preds30Tree60ADASYN_all = [preds30Tree60ADASYN_all; squeeze(preds30Tree60ADASYN)];
    
    [~,scores30Tree60ADASYN] = resubPredict((classification30Tree60ADASYN));
    [x30Tree60ADASYN,y30Tree60ADASYN,~,auc30Tree60ADASYN] = perfcurve(trainingData(train,end),scores30Tree60ADASYN(:,2),1);
    t14 = toc;
    
    %save('classification30Tree60ADASYN.mat','classification30Tree60ADASYN','-v7.3');
    save('classification30Tree60ADASYN_30.mat','classification30Tree60ADASYN','-v7.3');
    %save('classification30Tree60ADASYN_20.mat','classification30Tree60ADASYN','-v7.3');
    %save('classification30Tree60ADASYN_10.mat','classification30Tree60ADASYN','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree60ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree60ADASYN,~] = predict(classification40Tree60ADASYN,trainingData(test,1:end-1));
    targets40Tree60ADASYN = trainingData(test,end);
    targets40Tree60ADASYN_all = [targets40Tree60ADASYN_all; squeeze(targets40Tree60ADASYN)];
    preds40Tree60ADASYN_all = [preds40Tree60ADASYN_all; squeeze(preds40Tree60ADASYN)];
    
    [~,scores40Tree60ADASYN] = resubPredict((classification40Tree60ADASYN));
    [x40Tree60ADASYN,y40Tree60ADASYN,~,auc40Tree60ADASYN] = perfcurve(trainingData(train,end),scores40Tree60ADASYN(:,2),1);
    t15 = toc;
    
    %save('classification40Tree60ADASYN.mat','classification40Tree60ADASYN','-v7.3');
    save('classification40Tree60ADASYN_30.mat','classification40Tree60ADASYN','-v7.3');
    %save('classification40Tree60ADASYN_20.mat','classification40Tree60ADASYN','-v7.3');
    %save('classification40Tree60ADASYN_10.mat','classification40Tree60ADASYN','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree60ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree60ADASYN,~] = predict(classification50Tree60ADASYN,trainingData(test,1:end-1));
    targets50Tree60ADASYN = trainingData(test,end);
    targets50Tree60ADASYN_all = [targets50Tree60ADASYN_all; squeeze(targets50Tree60ADASYN)];
    preds50Tree60ADASYN_all = [preds50Tree60ADASYN_all; squeeze(preds50Tree60ADASYN)];
    
    [~,scores50Tree60ADASYN] = resubPredict((classification50Tree60ADASYN));
    [x50Tree60ADASYN,y50Tree60ADASYN,~,auc50Tree60ADASYN] = perfcurve(trainingData(train,end),scores50Tree60ADASYN(:,2),1);
    t16 = toc;
    
    %save('classification50Tree60ADASYN.mat','classification50Tree60ADASYN','-v7.3');
    save('classification50Tree60ADASYN_30.mat','classification50Tree60ADASYN','-v7.3');
    %save('classification50Tree60ADASYN_20.mat','classification50Tree60ADASYN','-v7.3');
    %save('classification50Tree60ADASYN_10.mat','classification50Tree60ADASYN','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM60ADASYN_all,predsLinSVM60ADASYN_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM60ADASYN_all,predsQuadSVM60ADASYN_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM60ADASYN_all,predsCubicSVM60ADASYN_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM60ADASYN_all,predsFineSVM60ADASYN_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM60ADASYN_all,predsMediumSVM60ADASYN_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM60ADASYN_all,predsCoarseSVM60ADASYN_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN60ADASYN_all,preds3KNN60ADASYN_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN60ADASYN_all,preds5KNN60ADASYN_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN60ADASYN_all,preds7KNN60ADASYN_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN60ADASYN_all,preds9KNN60ADASYN_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree60ADASYN_all,preds5Tree60ADASYN_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree60ADASYN_all,preds10Tree60ADASYN_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree60ADASYN_all,preds20Tree60ADASYN_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree60ADASYN_all,preds30Tree60ADASYN_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree60ADASYN_all,preds40Tree60ADASYN_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree60ADASYN_all,preds50Tree60ADASYN_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM60ADASYN,yLinSVM60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM60ADASYN,yQuadSVM60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM60ADASYN,yCubicSVM60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM60ADASYN,yFineSVM60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM60ADASYN,yMediumSVM60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM60ADASYN,yCoarseSVM60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN60ADASYN,y3KNN60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN60ADASYN,y5KNN60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN60ADASYN,y7KNN60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN60ADASYN,y9KNN60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree60ADASYN,y5Tree60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree60ADASYN,y10Tree60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree60ADASYN,y20Tree60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree60ADASYN,y30Tree60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree60ADASYN,y40Tree60ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree60ADASYN,y50Tree60ADASYN,'LineWidth',2)
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
AUC = [aucLinSVM60ADASYN;aucQuadSVM60ADASYN;aucCubicSVM60ADASYN;aucFineSVM60ADASYN;aucMediumSVM60ADASYN;...
    aucCoarseSVM60ADASYN;auc3KNN60ADASYN;auc5KNN60ADASYN;auc7KNN60ADASYN;auc9KNN60ADASYN;auc5Tree60ADASYN;...
    auc10Tree60ADASYN;auc20Tree60ADASYN;auc30Tree60ADASYN;auc40Tree60ADASYN;auc50Tree60ADASYN];

T = table(AUC,Time,Time_total);

