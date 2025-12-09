%% evaluate_improved_aug.m
clear; clc;
load('datastores_aug.mat','imdsTest','augTest');
load('humanTransferNet_aug.mat','netTransfer');

[predLabels,scores] = classify(netTransfer, augTest);
trueLabels = imdsTest.Labels;

accuracy = mean(predLabels == trueLabels) * 100;
fprintf('Improved model accuracy: %.2f%%\n', accuracy);

figure; confusionchart(trueLabels, predLabels);
title('Confusion matrix - improved model');

% Per-class metrics
classes = categories(trueLabels);
for i=1:numel(classes)
    cls = classes{i};
    TP = sum((predLabels==cls)&(trueLabels==cls));
    FP = sum((predLabels==cls)&(trueLabels~=cls));
    FN = sum((predLabels~=cls)&(trueLabels==cls));
    prec = TP/(TP+FP+eps);
    rec  = TP/(TP+FN+eps);
    f1 = 2*prec*rec/(prec+rec+eps);
    fprintf('Class %s: Precision=%.3f Recall=%.3f F1=%.3f\n', cls, prec, rec, f1);
end

% Save false positives for inspection
fpIdx = find((predLabels=='human') & (trueLabels~='human'));
fpFolder = fullfile(pwd,'false_positives');
if ~exist(fpFolder,'dir'), mkdir(fpFolder); end
for i=1:numel(fpIdx)
    I = readimage(imdsTest, fpIdx(i));
    imwrite(I, fullfile(fpFolder, sprintf('fp_%03d.jpg', i)));
end
fprintf('Saved %d false positives to %s for inspection\n', numel(fpIdx), fpFolder);
