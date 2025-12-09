%% show_random_predictions.m
% Display random predictions from the improved model

clear; clc;

% Load datastores and the trained network
load('datastores_aug.mat', 'imdsTest', 'inputSize');
load('humanTransferNet_aug.mat', 'netTransfer');

numSamples = 12;   % Set how many random predictions you want shown

% Get random indices
totalImages = numel(imdsTest.Files);
idx = randperm(totalImages, numSamples);

figure;
for i = 1:numSamples
    % Read image
    I = readimage(imdsTest, idx(i));

    % Resize image for the model
    Ir = imresize(I, inputSize(1:2));

    % Predict
    [predLabel, score] = classify(netTransfer, Ir);
    trueLabel = imdsTest.Labels(idx(i));

    % Confidence score
    [~, maxIdx] = max(score);
    conf = score(maxIdx) * 100;

    % Display
    subplot(3, 4, i);
    imshow(I);

    % Title color: green if correct, red if wrong
    if predLabel == trueLabel
        color = 'g';
    else
        color = 'r';
    end

    title(sprintf('Pred: %s (%.1f%%)\nTrue: %s', ...
        string(predLabel), conf, string(trueLabel)), ...
        'Color', color, 'FontSize', 11);
end

sgtitle('Random Example Predictions from Improved Model (94.44% Accuracy)', 'FontSize', 14, 'FontWeight', 'bold');
