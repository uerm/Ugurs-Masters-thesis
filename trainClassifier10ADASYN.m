function T = trainClassifier10ADASYN(trainingData)

indices = crossvalind('Kfold',trainingData(:,end),10);

%cp = classperf(response);
targetsLinSVM10ADASYN_all = [];
predsLinSVM10ADASYN_all = [];

targetsQuadSVM10ADASYN_all = [];
predsQuadSVM10ADASYN_all = [];

targetsCubicSVM10ADASYN_all = [];
predsCubicSVM10ADASYN_all = [];

targetsFineSVM10ADASYN_all = [];
predsFineSVM10ADASYN_all = [];

targetsMediumSVM10ADASYN_all = [];
predsMediumSVM10ADASYN_all = [];

targetsCoarseSVM10ADASYN_all = [];
predsCoarseSVM10ADASYN_all = [];

targets3KNN10ADASYN_all = [];
preds3KNN10ADASYN_all = [];

targets5KNN10ADASYN_all = [];
preds5KNN10ADASYN_all = [];

targets7KNN10ADASYN_all = [];
preds7KNN10ADASYN_all = [];

targets9KNN10ADASYN_all = [];
preds9KNN10ADASYN_all = [];

targets5Tree10ADASYN_all = [];
preds5Tree10ADASYN_all = [];

targets10Tree10ADASYN_all = [];
preds10Tree10ADASYN_all = [];

targets20Tree10ADASYN_all = [];
preds20Tree10ADASYN_all = [];

targets30Tree10ADASYN_all = [];
preds30Tree10ADASYN_all = [];

targets40Tree10ADASYN_all = [];
preds40Tree10ADASYN_all = [];

targets50Tree10ADASYN_all = [];
preds50Tree10ADASYN_all = [];


for i = 1:10
    test = (indices == i); 
    train = ~test;
    %class = classify(trainingData(test,1:end-1),trainingData(train,1:end-1),trainingData(train,end));
    
    % Linear SVM
    tic
    classificationLinearSVM10ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'linear', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsLinSVM10ADASYN,~] = predict(classificationLinearSVM10ADASYN,trainingData(test,1:end-1));
    targetLinSVM10ADASYN = trainingData(test,end);
    targetsLinSVM10ADASYN_all = [targetsLinSVM10ADASYN_all; squeeze(targetLinSVM10ADASYN)];
    predsLinSVM10ADASYN_all = [predsLinSVM10ADASYN_all; squeeze(predsLinSVM10ADASYN)];
    
    [~,scoresLinSVM10ADASYN] = resubPredict(fitPosterior(classificationLinearSVM10ADASYN));
    [xLinSVM10ADASYN,yLinSVM10ADASYN,~,aucLinSVM10ADASYN] = perfcurve(trainingData(train,end),scoresLinSVM10ADASYN(:,2),1);
    t1 = toc;
    
    %save('classificationLinearSVM10ADASYN.mat','classificationLinearSVM10ADASYN','-v7.3'); % majority voting
    %save('classificationLinearSVM10ADASYN_30.mat','classificationLinearSVM10ADASYN','-v7.3'); % Threshold 30%
    %save('classificationLinearSVM10ADASYN_20.mat','classificationLinearSVM10ADASYN','-v7.3'); % Threshold 20%
    save('classificationLinearSVM10ADASYN_10.mat','classificationLinearSVM10ADASYN','-v7.3'); % Threshold 10%
    

    
    %Quadratic SVM
    tic
    classificationQuadSVM10ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 2, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsQuadSVM10ADASYN,~] = predict(classificationQuadSVM10ADASYN,trainingData(test,1:end-1));
    targetQuadSVM10ADASYN = trainingData(test,end);
    targetsQuadSVM10ADASYN_all = [targetsQuadSVM10ADASYN_all; squeeze(targetQuadSVM10ADASYN)];
    predsQuadSVM10ADASYN_all = [predsQuadSVM10ADASYN_all; squeeze(predsQuadSVM10ADASYN)];
    
    [~,scoresQuadSVM10ADASYN] = resubPredict(fitPosterior(classificationQuadSVM10ADASYN));
    [xQuadSVM10ADASYN,yQuadSVM10ADASYN,~,aucQuadSVM10ADASYN] = perfcurve(trainingData(train,end),scoresQuadSVM10ADASYN(:,2),1);
    t2 = toc;
    
    %save('classificationQuadSVM10ADASYN.mat','classificationQuadSVM10ADASYN','-v7.3');
    %save('classificationQuadSVM10ADASYN_30.mat','classificationQuadSVM10ADASYN','-v7.3');
    %save('classificationQuadSVM10ADASYN_20.mat','classificationQuadSVM10ADASYN','-v7.3');
    save('classificationQuadSVM10ADASYN_10.mat','classificationQuadSVM10ADASYN','-v7.3');
    
    
    %Cubic SVM
    tic
    classificationCubicSVM10ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'polynomial', ...
    'PolynomialOrder', 3, ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCubicSVM10ADASYN,~] = predict(classificationCubicSVM10ADASYN,trainingData(test,1:end-1));
    targetCubicSVM10ADASYN = trainingData(test,end);
    targetsCubicSVM10ADASYN_all = [targetsCubicSVM10ADASYN_all; squeeze(targetCubicSVM10ADASYN)];
    predsCubicSVM10ADASYN_all = [predsCubicSVM10ADASYN_all; squeeze(predsCubicSVM10ADASYN)];
    
    [~,scoresCubicSVM10ADASYN] = resubPredict(fitPosterior(classificationCubicSVM10ADASYN));
    [xCubicSVM10ADASYN,yCubicSVM10ADASYN,~,aucCubicSVM10ADASYN] = perfcurve(trainingData(train,end),scoresCubicSVM10ADASYN(:,2),1);
    t3 = toc;
    
    %save('classificationCubicSVM10ADASYN.mat','classificationCubicSVM10ADASYN','-v7.3');
    %save('classificationCubicSVM10ADASYN_30.mat','classificationCubicSVM10ADASYN','-v7.3');
    %save('classificationCubicSVM10ADASYN_20.mat','classificationCubicSVM10ADASYN','-v7.3');
    save('classificationCubicSVM10ADASYN_10.mat','classificationCubicSVM10ADASYN','-v7.3');
    
    
    
    
    % Fine Gaussian SVM
    tic
    classificationFineSVM10ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 5.3, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsFineSVM10ADASYN,~] = predict(classificationFineSVM10ADASYN,trainingData(test,1:end-1));
    targetsFineSVM10ADASYN = trainingData(test,end);
    targetsFineSVM10ADASYN_all = [targetsFineSVM10ADASYN_all; squeeze(targetsFineSVM10ADASYN)];
    predsFineSVM10ADASYN_all = [predsFineSVM10ADASYN_all; squeeze(predsFineSVM10ADASYN)];
    
    [~,scoresFineSVM10ADASYN] = resubPredict(fitPosterior(classificationFineSVM10ADASYN));
    [xFineSVM10ADASYN,yFineSVM10ADASYN,~,aucFineSVM10ADASYN] = perfcurve(trainingData(train,end),scoresFineSVM10ADASYN(:,2),1);
    t4 = toc;
    
    %save('classificationFineSVM10ADASYN.mat','classificationFineSVM10ADASYN','-v7.3');
    %save('classificationFineSVM10ADASYN_30.mat','classificationFineSVM10ADASYN','-v7.3');
    %save('classificationFineSVM10ADASYN_20.mat','classificationFineSVM10ADASYN','-v7.3');
    save('classificationFineSVM10ADASYN_10.mat','classificationFineSVM10ADASYN','-v7.3');
    
    
    
    
    % Medium Gaussian SVM
    tic
    classificationMediumSVM10ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 21, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsMediumSVM10ADASYN,~] = predict(classificationMediumSVM10ADASYN,trainingData(test,1:end-1));
    targetsMediumSVM10ADASYN = trainingData(test,end);
    targetsMediumSVM10ADASYN_all = [targetsMediumSVM10ADASYN_all; squeeze(targetsMediumSVM10ADASYN)];
    predsMediumSVM10ADASYN_all = [predsMediumSVM10ADASYN_all; squeeze(predsMediumSVM10ADASYN)];
    
    [~,scoresMediumSVM10ADASYN] = resubPredict(fitPosterior(classificationMediumSVM10ADASYN));
    [xMediumSVM10ADASYN,yMediumSVM10ADASYN,~,aucMediumSVM10ADASYN] = perfcurve(trainingData(train,end),scoresMediumSVM10ADASYN(:,2),1);
    t5 = toc;
    
    %save('classificationMediumSVM10ADASYN.mat','classificationMediumSVM10ADASYN','-v7.3');
    %save('classificationMediumSVM10ADASYN_30.mat','classificationMediumSVM10ADASYN','-v7.3');
    %save('classificationMediumSVM10ADASYN_20.mat','classificationMediumSVM10ADASYN','-v7.3');
    save('classificationMediumSVM10ADASYN_10.mat','classificationMediumSVM10ADASYN','-v7.3');
    
    
    % Coarse Gaussian SVM
    tic
    classificationCoarseSVM10ADASYN = fitcsvm(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 85, ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [predsCoarseSVM10ADASYN,~] = predict(classificationCoarseSVM10ADASYN,trainingData(test,1:end-1));
    targetsCoarseSVM10ADASYN = trainingData(test,end);
    targetsCoarseSVM10ADASYN_all = [targetsCoarseSVM10ADASYN_all; squeeze(targetsCoarseSVM10ADASYN)];
    predsCoarseSVM10ADASYN_all = [predsCoarseSVM10ADASYN_all; squeeze(predsCoarseSVM10ADASYN)];
    
    [~,scoresCoarseSVM10ADASYN] = resubPredict(fitPosterior(classificationCoarseSVM10ADASYN));
    [xCoarseSVM10ADASYN,yCoarseSVM10ADASYN,~,aucCoarseSVM10ADASYN] = perfcurve(trainingData(train,end),scoresCoarseSVM10ADASYN(:,2),1);
    t6 = toc;
    
    %save('classificationCoarseSVM10ADASYN.mat','classificationCoarseSVM10ADASYN','-v7.3');
    %save('classificationCoarseSVM10ADASYN_30.mat','classificationCoarseSVM10ADASYN','-v7.3');
    %save('classificationCoarseSVM10ADASYN_20.mat','classificationCoarseSVM10ADASYN','-v7.3');
    save('classificationCoarseSVM10ADASYN_10.mat','classificationCoarseSVM10ADASYN','-v7.3');
    
    

    % kNN 3 neighbors
    tic
    classification3KNN10ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 3, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds3KNN10ADASYN,~] = predict(classification3KNN10ADASYN,trainingData(test,1:end-1));
    targets3KNN10ADASYN = trainingData(test,end);
    targets3KNN10ADASYN_all = [targets3KNN10ADASYN_all; squeeze(targets3KNN10ADASYN)];
    preds3KNN10ADASYN_all = [preds3KNN10ADASYN_all; squeeze(preds3KNN10ADASYN)];
    
    [~,scores3KNN10ADASYN] = resubPredict((classification3KNN10ADASYN));
    [x3KNN10ADASYN,y3KNN10ADASYN,~,auc3KNN10ADASYN] = perfcurve(trainingData(train,end),scores3KNN10ADASYN(:,2),1);
    t7 = toc;
    
    %save('classification3KNN10ADASYN.mat','classification3KNN10ADASYN','-v7.3');
    %save('classification3KNN10ADASYN_30.mat','classification3KNN10ADASYN','-v7.3');
    %save('classification3KNN10ADASYN_20.mat','classification3KNN10ADASYN','-v7.3');
    save('classification3KNN10ADASYN_10.mat','classification3KNN10ADASYN','-v7.3');
    
    
    
    
    
    % kNN 5 neighbors
    tic
    classification5KNN10ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 5, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds5KNN10ADASYN,~] = predict(classification5KNN10ADASYN,trainingData(test,1:end-1));
    targets5KNN10ADASYN = trainingData(test,end);
    targets5KNN10ADASYN_all = [targets5KNN10ADASYN_all; squeeze(targets5KNN10ADASYN)];
    preds5KNN10ADASYN_all = [preds5KNN10ADASYN_all; squeeze(preds5KNN10ADASYN)];
    
    [~,scores5KNN10ADASYN] = resubPredict((classification5KNN10ADASYN));
    [x5KNN10ADASYN,y5KNN10ADASYN,~,auc5KNN10ADASYN] = perfcurve(trainingData(train,end),scores5KNN10ADASYN(:,2),1);
    t8 = toc;

    %save('classification5KNN10ADASYN.mat','classification5KNN10ADASYN','-v7.3');
    %save('classification5KNN10ADASYN_30.mat','classification5KNN10ADASYN','-v7.3');
    %save('classification5KNN10ADASYN_20.mat','classification5KNN10ADASYN','-v7.3');
    save('classification5KNN10ADASYN_10.mat','classification5KNN10ADASYN','-v7.3');
    
    
    
    
    % kNN 7 neighbors
    tic
    classification7KNN10ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 7, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds7KNN10ADASYN,~] = predict(classification7KNN10ADASYN,trainingData(test,1:end-1));
    targets7KNN10ADASYN = trainingData(test,end);
    targets7KNN10ADASYN_all = [targets7KNN10ADASYN_all; squeeze(targets7KNN10ADASYN)];
    preds7KNN10ADASYN_all = [preds7KNN10ADASYN_all; squeeze(preds7KNN10ADASYN)];
    
    [~,scores7KNN10ADASYN] = resubPredict((classification7KNN10ADASYN));
    [x7KNN10ADASYN,y7KNN10ADASYN,~,auc7KNN10ADASYN] = perfcurve(trainingData(train,end),scores7KNN10ADASYN(:,2),1);
    t9 = toc;
    
    %save('classification7KNN10ADASYN.mat','classification7KNN10ADASYN','-v7.3');
    %save('classification7KNN10ADASYN_30.mat','classification7KNN10ADASYN','-v7.3');
    %save('classification7KNN10ADASYN_20.mat','classification7KNN10ADASYN','-v7.3');
    save('classification7KNN10ADASYN_10.mat','classification7KNN10ADASYN','-v7.3');
    
    
    
    
    % kNN 9 neighbors
    tic
    classification9KNN10ADASYN = fitcknn(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'Distance', 'Euclidean', ...
    'Exponent', [], ...
    'NumNeighbors', 9, ...
    'DistanceWeight', 'Equal', ...
    'Standardize', true, ...
    'ClassNames', [0; 1]);
    [preds9KNN10ADASYN,~] = predict(classification9KNN10ADASYN,trainingData(test,1:end-1));
    targets9KNN10ADASYN = trainingData(test,end);
    targets9KNN10ADASYN_all = [targets9KNN10ADASYN_all; squeeze(targets9KNN10ADASYN)];
    preds9KNN10ADASYN_all = [preds9KNN10ADASYN_all; squeeze(preds9KNN10ADASYN)];
    
    [~,scores9KNN10ADASYN] = resubPredict((classification9KNN10ADASYN));
    [x9KNN10ADASYN,y9KNN10ADASYN,~,auc9KNN10ADASYN] = perfcurve(trainingData(train,end),scores9KNN10ADASYN(:,2),1);
    t10 = toc;
    
    %save('classification9KNN10ADASYN.mat','classification9KNN10ADASYN','-v7.3');
    %save('classification9KNN10ADASYN_30.mat','classification9KNN10ADASYN','-v7.3');
    %save('classification9KNN10ADASYN_20.mat','classification9KNN10ADASYN','-v7.3');
    save('classification9KNN10ADASYN_10.mat','classification9KNN10ADASYN','-v7.3');
    

    % 5 split RF
    tic
    classification5Tree10ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 5, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds5Tree10ADASYN,~] = predict(classification5Tree10ADASYN,trainingData(test,1:end-1));
    targets5Tree10ADASYN = trainingData(test,end);
    targets5Tree10ADASYN_all = [targets5Tree10ADASYN_all; squeeze(targets5Tree10ADASYN)];
    preds5Tree10ADASYN_all = [preds5Tree10ADASYN_all; squeeze(preds5Tree10ADASYN)];
    
    [~,scores5Tree10ADASYN] = resubPredict((classification5Tree10ADASYN));
    [x5Tree10ADASYN,y5Tree10ADASYN,~,auc5Tree10ADASYN] = perfcurve(trainingData(train,end),scores5Tree10ADASYN(:,2),1);
    t11 = toc;
    
    %save('classification5Tree10ADASYN.mat','classification5Tree10ADASYN','-v7.3');
    %save('classification5Tree10ADASYN_30.mat','classification5Tree10ADASYN','-v7.3');
    %save('classification5Tree10ADASYN_20.mat','classification5Tree10ADASYN','-v7.3');
    save('classification5Tree10ADASYN_10.mat','classification5Tree10ADASYN','-v7.3');

    
    % 10 split RF
    tic
    classification10Tree10ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 10, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds10Tree10ADASYN,~] = predict(classification10Tree10ADASYN,trainingData(test,1:end-1));
    targets10Tree10ADASYN = trainingData(test,end);
    targets10Tree10ADASYN_all = [targets10Tree10ADASYN_all; squeeze(targets10Tree10ADASYN)];
    preds10Tree10ADASYN_all = [preds10Tree10ADASYN_all; squeeze(preds10Tree10ADASYN)];
    
    [~,scores10Tree10ADASYN] = resubPredict((classification10Tree10ADASYN));
    [x10Tree10ADASYN,y10Tree10ADASYN,~,auc10Tree10ADASYN] = perfcurve(trainingData(train,end),scores10Tree10ADASYN(:,2),1);
    t12 = toc;
    
    %save('classification10Tree10ADASYN.mat','classification10Tree10ADASYN','-v7.3');
    %save('classification10Tree10ADASYN_30.mat','classification10Tree10ADASYN','-v7.3');
    %save('classification10Tree10ADASYN_20.mat','classification10Tree10ADASYN','-v7.3');
    save('classification10Tree10ADASYN_10.mat','classification10Tree10ADASYN','-v7.3');
    
    
    % 20 split RF
    tic
    classification20Tree10ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 20, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds20Tree10ADASYN,~] = predict(classification20Tree10ADASYN,trainingData(test,1:end-1));
    targets20Tree10ADASYN = trainingData(test,end);
    targets20Tree10ADASYN_all = [targets20Tree10ADASYN_all; squeeze(targets20Tree10ADASYN)];
    preds20Tree10ADASYN_all = [preds20Tree10ADASYN_all; squeeze(preds20Tree10ADASYN)];
    [~,scores20Tree10ADASYN] = resubPredict((classification20Tree10ADASYN));
    
    [x20Tree10ADASYN,y20Tree10ADASYN,~,auc20Tree10ADASYN] = perfcurve(trainingData(train,end),scores20Tree10ADASYN(:,2),1);
    t13 = toc;
    
    %save('classification20Tree10ADASYN.mat','classification20Tree10ADASYN','-v7.3');
    %save('classification20Tree10ADASYN_30.mat','classification20Tree10ADASYN','-v7.3');
    %save('classification20Tree10ADASYN_20.mat','classification20Tree10ADASYN','-v7.3');
    save('classification20Tree10ADASYN_10.mat','classification20Tree10ADASYN','-v7.3');

    
    % 30 split RF
    tic
    classification30Tree10ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 30, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds30Tree10ADASYN,~] = predict(classification30Tree10ADASYN,trainingData(test,1:end-1));
    targets30Tree10ADASYN = trainingData(test,end);
    targets30Tree10ADASYN_all = [targets30Tree10ADASYN_all; squeeze(targets30Tree10ADASYN)];
    preds30Tree10ADASYN_all = [preds30Tree10ADASYN_all; squeeze(preds30Tree10ADASYN)];
    
    [~,scores30Tree10ADASYN] = resubPredict((classification30Tree10ADASYN));
    [x30Tree10ADASYN,y30Tree10ADASYN,~,auc30Tree10ADASYN] = perfcurve(trainingData(train,end),scores30Tree10ADASYN(:,2),1);
    t14 = toc;
    
    %save('classification30Tree10ADASYN.mat','classification30Tree10ADASYN','-v7.3');
    %save('classification30Tree10ADASYN_30.mat','classification30Tree10ADASYN','-v7.3');
    %save('classification30Tree10ADASYN_20.mat','classification30Tree10ADASYN','-v7.3');
    save('classification30Tree10ADASYN_10.mat','classification30Tree10ADASYN','-v7.3');
    

    
    % 40 split RF
    tic
    classification40Tree10ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 40, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds40Tree10ADASYN,~] = predict(classification40Tree10ADASYN,trainingData(test,1:end-1));
    targets40Tree10ADASYN = trainingData(test,end);
    targets40Tree10ADASYN_all = [targets40Tree10ADASYN_all; squeeze(targets40Tree10ADASYN)];
    preds40Tree10ADASYN_all = [preds40Tree10ADASYN_all; squeeze(preds40Tree10ADASYN)];
    
    [~,scores40Tree10ADASYN] = resubPredict((classification40Tree10ADASYN));
    [x40Tree10ADASYN,y40Tree10ADASYN,~,auc40Tree10ADASYN] = perfcurve(trainingData(train,end),scores40Tree10ADASYN(:,2),1);
    t15 = toc;
    
    %save('classification40Tree10ADASYN.mat','classification40Tree10ADASYN','-v7.3');
    %save('classification40Tree10ADASYN_30.mat','classification40Tree10ADASYN','-v7.3');
    %save('classification40Tree10ADASYN_20.mat','classification40Tree10ADASYN','-v7.3');
    save('classification40Tree10ADASYN_10.mat','classification40Tree10ADASYN','-v7.3');
    

    % 50 split RF
    tic
    classification50Tree10ADASYN = fitctree(...
    trainingData(train,1:end-1),...
    trainingData(train,end), ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 50, ...
    'Surrogate', 'off', ...
    'ClassNames', [0; 1]);
    [preds50Tree10ADASYN,~] = predict(classification50Tree10ADASYN,trainingData(test,1:end-1));
    targets50Tree10ADASYN = trainingData(test,end);
    targets50Tree10ADASYN_all = [targets50Tree10ADASYN_all; squeeze(targets50Tree10ADASYN)];
    preds50Tree10ADASYN_all = [preds50Tree10ADASYN_all; squeeze(preds50Tree10ADASYN)];
    
    [~,scores50Tree10ADASYN] = resubPredict((classification50Tree10ADASYN));
    [x50Tree10ADASYN,y50Tree10ADASYN,~,auc50Tree10ADASYN] = perfcurve(trainingData(train,end),scores50Tree10ADASYN(:,2),1);
    t16 = toc;
    
    %save('classification50Tree10ADASYN.mat','classification50Tree10ADASYN','-v7.3');
    %save('classification50Tree10ADASYN_30.mat','classification50Tree10ADASYN','-v7.3');
    %save('classification50Tree10ADASYN_20.mat','classification50Tree10ADASYN','-v7.3');
    save('classification50Tree10ADASYN_10.mat','classification50Tree10ADASYN','-v7.3');
    

    %valAcc = classperf(cp,class,test);
end
%sum(preds_all==targets_all)/length(preds_all);

%confusionmat(targetsFineSVM_all,predsFineSVM_all)


% Confusion matrices
tic
figure()
confusionchart(targetsLinSVM10ADASYN_all,predsLinSVM10ADASYN_all)
title('Linear SVM')
t17 = toc;

tic
figure()
confusionchart(targetsQuadSVM10ADASYN_all,predsQuadSVM10ADASYN_all)
title('Quadratic SVM')
t18 = toc;

tic
figure()
confusionchart(targetsCubicSVM10ADASYN_all,predsCubicSVM10ADASYN_all)
title('Cubic SVM')
t19 = toc;

tic
figure()
confusionchart(targetsFineSVM10ADASYN_all,predsFineSVM10ADASYN_all)
title('Fine Gaussian SVM')
t20 = toc;

tic
figure()
confusionchart(targetsMediumSVM10ADASYN_all,predsMediumSVM10ADASYN_all)
title('Medium Gaussian SVM')
t21 = toc;

tic
figure()
confusionchart(targetsCoarseSVM10ADASYN_all,predsCoarseSVM10ADASYN_all)
title('Coarse Gaussian SVM')
t22 = toc;

tic
figure()
confusionchart(targets3KNN10ADASYN_all,preds3KNN10ADASYN_all)
title('k = 3')
t23 = toc;

tic
figure()
confusionchart(targets5KNN10ADASYN_all,preds5KNN10ADASYN_all)
title('k = 5')
t24 = toc;


tic
figure()
confusionchart(targets7KNN10ADASYN_all,preds7KNN10ADASYN_all)
title('k = 7')
t25 = toc;

tic
figure()
confusionchart(targets9KNN10ADASYN_all,preds9KNN10ADASYN_all)
title('k = 9')
t26 = toc;



tic
figure()
confusionchart(targets5Tree10ADASYN_all,preds5Tree10ADASYN_all)
title('RF, 5 Trees')
t27 = toc;


tic
figure()
confusionchart(targets10Tree10ADASYN_all,preds10Tree10ADASYN_all)
title('RF, 10 Trees')
t28 = toc;


tic
figure()
confusionchart(targets20Tree10ADASYN_all,preds20Tree10ADASYN_all)
title('RF, 20 Trees')
t29 = toc;


tic
figure()
confusionchart(targets30Tree10ADASYN_all,preds30Tree10ADASYN_all)
title('RF, 30 Trees')
t30 = toc;


tic
figure()
confusionchart(targets40Tree10ADASYN_all,preds40Tree10ADASYN_all)
title('RF, 40 Trees')
t31 = toc;

tic
figure()
confusionchart(targets50Tree10ADASYN_all,preds50Tree10ADASYN_all)
title('RF, 50 Trees')
t32 = toc;



% ROC curve plots, SVM
tic
figure()
plot(xLinSVM10ADASYN,yLinSVM10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Linear SVM')
t33 = toc;

tic
figure()
plot(xQuadSVM10ADASYN,yQuadSVM10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Quadratic SVM')
t34 = toc;


tic
figure()
plot(xCubicSVM10ADASYN,yCubicSVM10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Cubic SVM')
t35 = toc;


tic
figure()
plot(xFineSVM10ADASYN,yFineSVM10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Fine Gaussian SVM')
t36 = toc;


tic
figure()
plot(xMediumSVM10ADASYN,yMediumSVM10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Medium Gaussian SVM')
t37 = toc;


tic
figure()
plot(xCoarseSVM10ADASYN,yCoarseSVM10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, Coarse Gaussian SVM')
t38 = toc;


% kNNs
tic
figure()
plot(x3KNN10ADASYN,y3KNN10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 3')
t39 = toc;


tic
figure()
plot(x5KNN10ADASYN,y5KNN10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 5')
t40 = toc;


tic
figure()
plot(x7KNN10ADASYN,y7KNN10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 7')
t41 = toc;


tic
figure()
plot(x9KNN10ADASYN,y9KNN10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, k = 9')
t42 = toc;


tic
% Random Forest
figure()
plot(x5Tree10ADASYN,y5Tree10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 5 Trees')
t43 = toc;


tic
figure()
plot(x10Tree10ADASYN,y10Tree10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 10 Trees')
t44 = toc;


tic
figure()
plot(x20Tree10ADASYN,y20Tree10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 20 Trees')
t45 = toc;


tic
figure()
plot(x30Tree10ADASYN,y30Tree10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 30 Trees')
t46 = toc;


tic
figure()
plot(x40Tree10ADASYN,y40Tree10ADASYN,'LineWidth',2)
xlabel('False Positive Rate')
ylabel('True Positive Rate')
title('ROC, 40 Trees')
t47 = toc;


tic
figure()
plot(x50Tree10ADASYN,y50Tree10ADASYN,'LineWidth',2)
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
AUC = [aucLinSVM10ADASYN;aucQuadSVM10ADASYN;aucCubicSVM10ADASYN;aucFineSVM10ADASYN;aucMediumSVM10ADASYN;...
    aucCoarseSVM10ADASYN;auc3KNN10ADASYN;auc5KNN10ADASYN;auc7KNN10ADASYN;auc9KNN10ADASYN;auc5Tree10ADASYN;...
    auc10Tree10ADASYN;auc20Tree10ADASYN;auc30Tree10ADASYN;auc40Tree10ADASYN;auc50Tree10ADASYN];

T = table(AUC,Time,Time_total);

