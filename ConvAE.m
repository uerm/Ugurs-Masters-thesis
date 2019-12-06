latentDim = 50;

encoderLG = layerGraph([
    imageInputLayer([1 1100],'Name','input_encoder','Normalization','none')
    convolution2dLayer([1 50], 32, 'Padding','same', 'Stride', [1,10], 'Name', 'conv1')
    reluLayer('Name','relu1')
    convolution2dLayer([1 50], 64, 'Padding','same', 'Stride', [1,10], 'Name', 'conv2')
    reluLayer('Name','relu2')
    convolution2dLayer([1 50], 64, 'Padding','same', 'Stride', [1,10], 'Name', 'conv3')
    reluLayer('Name','relu3')
    convolution2dLayer([1 50], 64, 'Padding','same', 'Stride', [1,10], 'Name', 'conv4')
    reluLayer('Name','relu4')
    convolution2dLayer([1 50], 64, 'Padding','same', 'Stride', [1,10], 'Name', 'conv5')
    reluLayer('Name','relu5')
    fullyConnectedLayer(2 * latentDim, 'Name', 'fc_encoder')
    ]);

decoderLG = layerGraph([
    imageInputLayer([1 1 latentDim],'Name','i','Normalization','none')
    transposedConv2dLayer([1 50], 64, 'Cropping', 'same', 'Stride', [1,11], 'Name', 'transpose1')
    reluLayer('Name','relu1')
    transposedConv2dLayer([1 50], 64, 'Cropping', 'same', 'Stride', [1,5], 'Name', 'transpose2')
    reluLayer('Name','relu2')
    transposedConv2dLayer([1 50], 32, 'Cropping', 'same', 'Stride', [1,5], 'Name', 'transpose3')
    reluLayer('Name','relu3')
    transposedConv2dLayer([1 50], 32, 'Cropping', 'same', 'Stride', [1,2], 'Name', 'transpose4')
    reluLayer('Name','relu4')
    transposedConv2dLayer([1 50], 32, 'Cropping', 'same', 'Stride', [1,2], 'Name', 'transpose5')
    reluLayer('Name','relu5')
    transposedConv2dLayer([1 1100], 1, 'Cropping', 'same', 'Name', 'transpose6')
    ]);

encoderNet = dlnetwork(encoderLG);
decoderNet = dlnetwork(decoderLG);

executionEnvironment = "auto";

XTrain = sineTrain;
XTest = sineTest;
XTest = dlarray(XTest, 'SSCB');

numTrainImages = 100;

numEpochs = 50;
miniBatchSize = 512;
lr = 1e-3;
numIterations = floor(numTrainImages/miniBatchSize);
iteration = 0;

avgGradientsEncoder = [];
avgGradientsSquaredEncoder = [];
avgGradientsDecoder = [];
avgGradientsSquaredDecoder = [];

for epoch = 1:numEpochs
    tic;
    for i = 1:numIterations
        iteration = iteration + 1;
        idx = (i-1)*miniBatchSize+1:i*miniBatchSize;
        XBatch = XTrain(:,:,:,idx);
        XBatch = dlarray(XBatch, 'SSCB');
        
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XBatch = gpuArray(XBatch);           
        end 
            
        [infGrad, genGrad] = dlfeval(...
            @modelGradients, encoderNet, decoderNet, XBatch);
        
        [decoderNet.Learnables, avgGradientsDecoder, avgGradientsSquaredDecoder] = ...
            adamupdate(decoderNet.Learnables, ...
                genGrad, avgGradientsDecoder, avgGradientsSquaredDecoder, iteration, lr);
        [encoderNet.Learnables, avgGradientsEncoder, avgGradientsSquaredEncoder] = ...
            adamupdate(encoderNet.Learnables, ...
                infGrad, avgGradientsEncoder, avgGradientsSquaredEncoder, iteration, lr);
    end
    elapsedTime = toc;
    
    [z, zMean, zLogvar] = sampling(encoderNet, XTest);
    xPred = sigmoid(forward(decoderNet, z));
    elbo = ELBOloss(XTest, xPred, zMean, zLogvar);
    disp("Epoch : "+epoch+" Test ELBO loss = "+gather(extractdata(elbo))+...
        ". Time taken for epoch = "+ elapsedTime + "s")    
end
%%
xPlot = extractdata(xPred);
plot(xPlot(1,:,1,1))
hold on
plot(sineTest(1,:,1,1))




%%
z = reconstruction(encoderNet, XTest);
xPlot = sigmoid(forward(decoderNet, z));
xPlot = extractdata(xPlot);
plot(xPlot(1,:,1,1))
hold on
plot(sineTest(1,:,1,1))