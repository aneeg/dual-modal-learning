%%
% clc
% clear

%imagesReziseGoogleNet; % one
%% load
load('/data/ahmedfares/MatWork/801010/Data/catNameUniqueTrain');
load('/data/ahmedfares/MatWork/801010/Data/catNameUniqueVal');
load('/data/ahmedfares/MatWork/801010/Data/catNameUniqueTest');


load('/data/ahmedfares/MatWork/801010/Data/averageEncoderOutputTrain');
load('/data/ahmedfares/MatWork/801010/Data/averageEncoderOutputVal');
load('/data/ahmedfares/MatWork/801010/Data/averageEncoderOutputTest');

%% Train
imagesPath = fullfile(pwd,'Imagenet4');
trainingImagesInputs = cell(size(catNameUniqueTrain,1),1);
trainingImagesLabels = cell(size(catNameUniqueTrain,1),1);
for i=1 : size(catNameUniqueTrain,1)
    className = strsplit(catNameUniqueTrain{i,1},'_');
    trainingImagesInputs{i,1} = fullfile(imagesPath,className{1},erase(catNameUniqueTrain{i,1},'.eeg.mat'));
    temp= strsplit(className{1},'n');
    trainingImagesLabels{i,1} = temp {1,2}; % used in softmax
end

% Convert to 4-D Double
% Loop over filenames, inserting the image.
numTrainImages = numel(trainingImagesInputs);
array4dTrainImages = uint8(zeros(224, 224,3, numTrainImages));
for slice = 1 : numTrainImages
  [rows, columns, numberOfColorChannels] = size(imread(trainingImagesInputs{slice,1}));
  if numberOfColorChannels < 3
    message = 'Error: Image is not RGB Color!';
    uiwait(warndlg(message));
    continue;
  end
  if rows ~= 224 || columns ~= 224
    message = 'Error: Image is not 224x224!';
    uiwait(warndlg(message));
    continue; % Skip this image.
  end
  % Image is okay.  Insert it.
  array4dTrainImages(:,:,:,slice) = imread(trainingImagesInputs{slice,1});
end
save(fullfile(pwd,'DataGoogleNet','array4dTrainImages'),'array4dTrainImages', '-v7.3');
%% Val
valImagesInputs = cell(size(catNameUniqueVal,1),1);
valImagesLabels = cell(size(catNameUniqueVal,1),1);
for i=1 : size(catNameUniqueVal,1)
    className = strsplit(catNameUniqueVal{i,1},'_');
    valImagesInputs{i,1} = fullfile(imagesPath,className{1},erase(catNameUniqueVal{i,1},'.eeg.mat'));
    temp= strsplit(className{1},'n');
    valImagesLabels{i,1}= temp {1,2}; % used in softmax
end
%
numValImages = numel(valImagesInputs);
array4dValImages = uint8(zeros(224, 224,3, numValImages));
for slice = 1 : numValImages
  [rows, columns, numberOfColorChannels] = size(imread(valImagesInputs{slice,1}));
  if numberOfColorChannels < 3
    message = 'Error: Image is not RGB Color!';
    uiwait(warndlg(message));
    continue;
  end
  if rows ~= 224 || columns ~= 224
    message = 'Error: Image is not 224x224!';
    uiwait(warndlg(message));
    continue; % Skip this image.
  end
  % Image is okay.  Insert it.
  array4dValImages(:,:,:,slice) = imread(valImagesInputs{slice,1});
end
save(fullfile(pwd,'DataGoogleNet','array4dValImages'),'array4dValImages', '-v7.3');
%% Test
testImagesInputs = cell(size(catNameUniqueTest,1),1);
testImagesLabels = cell(size(catNameUniqueTest,1),1);
for i=1 : size(catNameUniqueTest,1)
    className = strsplit(catNameUniqueTest{i,1},'_');
    testImagesInputs{i,1} = fullfile(imagesPath,className{1},erase(catNameUniqueTest{i,1},'.eeg.mat'));
    temp= strsplit(className{1},'n');
    testImagesLabels{i,1}= temp {1,2}; % used in softmax
end
%
numTestImages = numel(testImagesInputs);
array4dTestImages = uint8(zeros(224, 224,3, numTestImages));
for slice = 1 : numTestImages
  [rows, columns, numberOfColorChannels] = size(imread(testImagesInputs{slice,1}));
  if numberOfColorChannels < 3
    message = 'Error: Image is not RGB Color!';
    uiwait(warndlg(message));
    continue;
  end
  if rows ~= 224 || columns ~= 224
    message = 'Error: Image is not 224x224!';
    uiwait(warndlg(message));
    continue; % Skip this image.
  end
  % Image is okay.  Insert it.
  array4dTestImages(:,:,:,slice) = imread(testImagesInputs{slice,1});
end
save(fullfile(pwd,'DataGoogleNet','array4dTestImages'),'array4dTestImages', '-v7.3');
%% EEG Feature representation 

matTrainingEEGFeaturesLabels =averageEncoderOutputTrain;
matValEEGFeaturesLabels = averageEncoderOutputVal;
matTestEEGFeaturesLabels = averageEncoderOutputTest;

%%
save(fullfile(pwd,'DataGoogleNet','trainImagesLabels'),'trainingImagesLabels', '-v7.3');
save(fullfile(pwd,'DataGoogleNet','valImagesLabels'),'valImagesLabels', '-v7.3');
save(fullfile(pwd,'DataGoogleNet','testImagesLabels'),'testImagesLabels', '-v7.3');

%%
function imagesReziseGoogleNet
mainFolder = fullfile(pwd,'Imagenet4');
foldersPaths = dir(fullfile(mainFolder));
for i = 3 : size(foldersPaths,1)
    subFolder=foldersPaths(i).name;
    filesPaths = dir(fullfile(foldersPaths(i).folder, subFolder,'*.JPEG'));
    for k = 1 : length(filesPaths)
        image = imread(fullfile(filesPaths(i).folder,filesPaths(k).name));
        I = imresize(image,[224,224]);
        if size(image,3)~= 3
            I = cat(3, I, I, I);
        end
        imwrite(I,fullfile(filesPaths(i).folder , filesPaths(k).name));
    end
end
end
