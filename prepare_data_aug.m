%% prepare_data_aug.m
% Prepare datastores from augmented dataset folder Dataset_aug

clear; clc;
datasetRoot = fullfile(pwd,'Dataset_aug'); % augmented data created above
if ~exist(datasetRoot,'dir')
    error('Dataset_aug folder not found. Run create_brightness_aug first.');
end

imds = imageDatastore(datasetRoot, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

tbl = countEachLabel(imds);
disp('Counts in augmented dataset:'); disp(tbl);

% Split
rng(1);
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

disp('Training counts:'); countEachLabel(imdsTrain)
disp('Testing counts:'); countEachLabel(imdsTest)

% Optional: oversample minority class by duplicating entries
T = countEachLabel(imdsTrain);
cnts = T.Count;
maxCnt = max(cnts);
files = imdsTrain.Files;
labels = imdsTrain.Labels;
for i = 1:height(T)
    lbl = T.Label(i);
    idx = find(labels==lbl);
    if T.Count(i) < maxCnt
        need = maxCnt - T.Count(i);
        reps = repmat(files(idx), ceil(need/numel(idx)), 1);
        reps = reps(1:need);
        files = [files; reps];
        labels = [labels; repmat(lbl, numel(reps), 1)];
    end
end

imdsTrain = imageDatastore(files, 'Labels', labels);
disp('After oversampling train counts:'); countEachLabel(imdsTrain)

% Create augmentedImageDatastore for training and testing
inputSize = [224 224 3];
imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-10 10], ...
    'RandXTranslation', [-8 8], ...
    'RandYTranslation', [-8 8], ...
    'RandScale', [0.95 1.05], ...
    'RandXReflection', true);

augTrain = augmentedImageDatastore(inputSize, imdsTrain, 'DataAugmentation', imageAugmenter, 'OutputSizeMode','resize');
augTest  = augmentedImageDatastore(inputSize, imdsTest, 'OutputSizeMode','resize');

save('datastores_aug.mat','imdsTrain','imdsTest','augTrain','augTest','inputSize');
disp('prepare_data_aug complete and datastores_aug.mat saved.');
