%% train_transfer_aug.m
clear; clc;
load('datastores_aug.mat','imdsTrain','imdsTest','augTrain','augTest','inputSize');

% Choose pretrained network
if exist('resnet18','file')
    baseNet = resnet18;
    fprintf('Using resnet18\n');
else
    baseNet = alexnet;
    fprintf('resnet18 not found, using alexnet\n');
end

% Build layer graph
lgraph = layerGraph(baseNet);

numClasses = numel(categories(imdsTrain.Labels));

% Replace final layers depending type
if isa(baseNet,'SeriesNetwork') % alexnet
    lgraph = removeLayers(lgraph, {'fc8','prob','output'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc_new','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax_new')
        classificationLayer('Name','classoutput_new')];
    lgraph = addLayers(lgraph, newLayers);
    lgraph = connectLayers(lgraph,'fc7','fc_new');
else
    lgraph = removeLayers(lgraph, {'fc1000','prob','ClassificationLayer_predictions'});
    newLayers = [
        fullyConnectedLayer(numClasses,'Name','fc_new','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
        softmaxLayer('Name','softmax_new')
        classificationLayer('Name','classoutput_new')];
    lgraph = addLayers(lgraph, newLayers);
    lgraph = connectLayers(lgraph,'pool5','fc_new');
end

% Training options
miniBatchSize = 32;
valFrequency = floor(numel(imdsTrain.Files)/miniBatchSize);
options = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize, ...
    'MaxEpochs', 12, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augTest, ...
    'ValidationFrequency', valFrequency, ...
    'Verbose', true, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment','auto', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',4);

% Train
netTransfer = trainNetwork(augTrain, lgraph, options);

% Save
save('humanTransferNet_aug.mat','netTransfer','lgraph','options');
disp('Training completed and saved to humanTransferNet_aug.mat');
