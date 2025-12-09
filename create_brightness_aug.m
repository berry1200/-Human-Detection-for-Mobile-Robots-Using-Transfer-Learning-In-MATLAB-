%% create_brightness_aug.m
% Create brightness- and contrast-augmented copies of training images.
% Run: create_brightness_aug

clear; clc;
rng(1);

origRoot = fullfile(pwd,'Dataset');  % your original dataset folder (human / nonhuman)
augRoot  = fullfile(pwd,'Dataset_aug'); % folder to create with extra images

if ~exist(origRoot,'dir')
    error('Original dataset folder "Dataset" not found in current folder.');
end

% Create augmented dataset root
if ~exist(augRoot,'dir')
    mkdir(augRoot);
end

labels = dir(origRoot);
labels = labels([labels.isdir] & ~ismember({labels.name},{'.','..'}));

% for each class (human, nonhuman)
for li = 1:numel(labels)
    lbl = labels(li).name;
    srcFolder = fullfile(origRoot, lbl);
    dstFolder = fullfile(augRoot, lbl);
    if ~exist(dstFolder,'dir'), mkdir(dstFolder); end
    
    files = dir(fullfile(srcFolder,'*.jpg'));
    files = [files; dir(fullfile(srcFolder,'*.png'))]; % add png
    fprintf('Processing class: %s (%d files)\n', lbl, numel(files));
    
    for k = 1:numel(files)
        fname = files(k).name;
        srcFile = fullfile(srcFolder,fname);
        img = im2single(imread(srcFile));
        % save original to new folder (keep same filename)
        imwrite(im2uint8(img), fullfile(dstFolder,fname));
        
        % create N augmented variants per image (you can change N)
        N = 3; % 3 variants per original
        for n = 1:N
            im = img;
            % random gamma (darken/brighten)
            gamma = 0.6 + rand()*1.4; % range ~[0.6,2.0]
            im2 = imadjust(im, [], [], gamma);
            
            % random contrast via adapthisteq (CLAHE) sometimes
            if rand() < 0.5
                % convert to grayscale for adapthisteq then back to rgb
                try
                    imGray = rgb2gray(im2);
                    imGray = adapthisteq(imGray);
                    % blend with original channels to keep color info
                    im2 = cat(3, imGray, imGray, imGray);
                catch
                    % if adapthisteq not available, skip
                end
            end
            
            % slight random rotation and crop
            ang = -8 + 16*rand();
            im2 = imrotate(im2, ang, 'bilinear', 'crop');
            
            % slight scale
            scale = 0.92 + rand()*0.16;
            im2 = imresize(im2, scale);
            im2 = imresize(im2, size(img(:,:,1)));
            
            % clip
            im2(im2>1) = 1; im2(im2<0) = 0;
            
            % save file with suffix
            [~,base,ext] = fileparts(fname);
            outname = sprintf('%s_aug%d%s', base, n, ext);
            imwrite(im2uint8(im2), fullfile(dstFolder,outname));
        end
    end
end

fprintf('Brightness augmentation complete. Augmented dataset saved to: %s\n', augRoot);
