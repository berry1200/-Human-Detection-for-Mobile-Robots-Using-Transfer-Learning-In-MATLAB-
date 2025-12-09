%% demo_image_file.m
% Demo: classify a single saved image using netTransfer

clear; clc; close all;

% Load trained network and input size
load('humanTransferNet_aug.mat','netTransfer');
if exist('inputSize','var') == 0
    % fallback: read from network if inputSize not stored
    inputSize = netTransfer.Layers(1).InputSize;
end

% ---- Set your image file here ----
imageFile = "C:\Cmet Semester 2\MATLAB\HumanDetection_MATLAB_Project\26.png";   % change to your image name/path
% ----------------------------------

% Check file existence
if ~exist(imageFile, 'file')
    error('Image file "%s" not found in current folder.', imageFile);
end

% Read image
I = imread(imageFile);

% Resize image to network input size
Ir = imresize(I, inputSize(1:2));

% Classify
[label, scores] = classify(netTransfer, Ir);
[conf, idx] = max(scores);
conf = conf * 100;   % convert to percentage

% Display in Command Window
fprintf('Predicted label: %s\n', string(label));
fprintf('Confidence: %.2f %%\n', conf);

% Display image with title
figure;
imshow(I);
title(sprintf('Pred: %s (%.2f%% confidence)', string(label), conf), ...
      'FontSize', 14, 'Interpreter','none');
