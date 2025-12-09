%% STEP 12: Full Evaluation of Human Detection Model
% Make sure these files exist in the current folder:
%   - datastores_aug.mat
%   - humanTransferNet_aug.mat

clear; clc;

%% 1) Load test data and trained network
load('datastores_aug.mat', 'imdsTest', 'inputSize');
load('humanTransferNet_aug.mat', 'netTransfer');

% Rebuild augTest (safe for all MATLAB versions)
augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'OutputSizeMode','resize');

fprintf('Loaded model and test set.\n');
fprintf('Number of test images: %d\n', numel(imdsTest.Files));

%% 2) Get predictions and scores
[predLabels, scores] = classify(netTransfer, augTest);
trueLabels = imdsTest.Labels;

%% 3) Overall accuracy
accuracy = mean(predLabels == trueLabels) * 100;
fprintf('\n=== OVERALL ACCURACY ===\n');
fprintf('Test accuracy: %.2f %%\n', accuracy);

%% 4) Confusion matrix
figure;
confusionchart(trueLabels, predLabels);
title(sprintf('Confusion Matrix (Accuracy = %.2f%%)', accuracy));

%% 5) Precision, Recall, F1-score per class
classes = categories(trueLabels);
numClasses = numel(classes);

precisionVec = zeros(numClasses,1);
recallVec    = zeros(numClasses,1);
f1Vec        = zeros(numClasses,1);

fprintf('\n=== CLASS-WISE METRICS ===\n');
for i = 1:numClasses
    cls = classes{i};
    TP = sum((predLabels == cls) & (trueLabels == cls));
    FP = sum((predLabels == cls) & (trueLabels ~= cls));
    FN = sum((predLabels ~= cls) & (trueLabels == cls));
    precision = TP / (TP + FP + eps);
    recall    = TP / (TP + FN + eps);
    f1        = 2*precision*recall / (precision + recall + eps);

    precisionVec(i) = precision;
    recallVec(i)    = recall;
    f1Vec(i)        = f1;

    fprintf('Class %-8s  Precision = %.3f  Recall = %.3f  F1 = %.3f\n', ...
        cls, precision, recall, f1);
end

% Optional: table of metrics
metricsTable = table(classes, precisionVec, recallVec, f1Vec, ...
    'VariableNames', {'Class', 'Precision', 'Recall', 'F1'});
disp(' ');
disp(metricsTable);

%% 6) ROC curve & AUC for "human" class
try
    % Find column index in 'scores' that corresponds to 'human'
    [~, idxHuman] = ismember(categorical("human"), netTransfer.Layers(end).Classes);
    humanScores = scores(:, idxHuman);

    % Create binary labels: 1 = human, 0 = nonhuman
    yTrue = double(trueLabels == categorical("human"));

    % perfcurve: labels (0/1), scores, positive class = 1
    [fpRate, tpRate, ~, AUC] = perfcurve(yTrue, humanScores, 1);

    figure;
    plot(fpRate, tpRate, 'LineWidth', 1.5);
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(sprintf('ROC Curve for class "human" (AUC = %.3f)', AUC));
    grid on;
catch ME
    warning('Could not compute ROC curve: %s', ME.message);
end

%% 7) Prediction speed (time per frame & FPS)
% We measure how long it takes to classify all test images.

% Rebuild augTest without shuffling
augTest = augmentedImageDatastore(inputSize, imdsTest, ...
    'OutputSizeMode','resize');

reset(augTest);  % start from first image

tic;
predTest = classify(netTransfer, augTest);
elapsedTime = toc;

numImages = numel(imdsTest.Files);
timePerImage = elapsedTime / numImages;   % seconds per image
FPS = 1 / timePerImage;

fprintf('\n=== RUNTIME PERFORMANCE ===\n');
fprintf('Total prediction time: %.3f s for %d images\n', elapsedTime, numImages);
fprintf('Average time per image: %.4f s (%.2f ms)\n', timePerImage, timePerImage*1000);
fprintf('Approx. processing speed: %.2f FPS\n', FPS);

%% DONE
disp('Step 12 evaluation completed.');
