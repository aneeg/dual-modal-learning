load('/data/ahmedfares/MatWork/801010/Data/averageEncoderOutputTrain');
imagesReziseSelected; %one
%%
tarinCounter=1;
valCounter=1;
testCounter = 1;


mainFolder = fullfile(pwd,'Selected18');
foldersPaths = dir(fullfile(mainFolder));
for i = 3 : size(foldersPaths,1) % number of classes
    className=foldersPaths(i).name; % subFolder-->class 
    imagesPath = dir(fullfile(foldersPaths(i).folder, className,'*.jpg'));
    numImages = length(imagesPath);
    if numImages> 40
        for j =1 : 40
           trainingImagesInputs{tarinCounter,1}= fullfile(imagesPath(i).folder,imagesPath(j).name);
           temp = strsplit(className,'n');
           trainingImagesLabels{tarinCounter,1} = temp{1,2};
           tarinCounter = tarinCounter+1;
        end
        x=round(((numImages-40)/2));
        if x > 5
            x=5;
        end
        for k=1 : x %val
            valImagesInputs{valCounter,1} = fullfile(imagesPath(i).folder,imagesPath(k+40).name);
            temp = strsplit(className,'n');
            valImagesLabels{valCounter,1}=temp{1,2};
            valCounter = valCounter+1;
        end 
        e = floor(((numImages-40)/2));
        if e >5
            e=5;
        end
        for m=1 : e
            testImagesInputs{testCounter,1} =  fullfile(imagesPath(i).folder,imagesPath(m+40+x).name);
            temp = strsplit(className,'n');
            testImagesLabels{testCounter,1}=temp{1,2};
            testCounter =testCounter+1;
        end
    else
        for k=1 : 5 %val
            valImagesInputs{valCounter,1} = fullfile(imagesPath(i).folder,imagesPath(k).name);
            temp = strsplit(className,'n');
            valImagesLabels{valCounter,1}=temp{1,2};
            valCounter = valCounter+1;
        end 
        for m=1 : 5
            testImagesInputs{testCounter,1} =  fullfile(imagesPath(i).folder,imagesPath(m+5).name);
            temp = strsplit(className,'n');
            testImagesLabels{testCounter,1}=temp{1,2};
            testCounter =testCounter+1;
        end
        for j =1 : numImages-10
           trainingImagesInputs{tarinCounter,1}= fullfile(imagesPath(i).folder,imagesPath(j+10).name);
           temp = strsplit(className,'n');
           trainingImagesLabels{tarinCounter,1} = temp{1,2};
           tarinCounter = tarinCounter+1;
        end
    end 

end
load(fullfile(pwd,'Cal/18/orderTrain'));
load(fullfile(pwd,'Cal/18/orderTest'));
load(fullfile(pwd,'Cal/18/orderVal'));
% orderTrain = randperm(692)'; %692 -->18, 17---> 652
% orderTest = randperm (80)'; %80 -->18, 17--->85
% orderVal = randperm (80)'; %80 -->18, 17 --->85
trainingImagesInputs= trainingImagesInputs(orderTrain(1:692),:);
trainingImagesLabels = trainingImagesLabels(orderTrain(1:692),:);

valImagesInputs= valImagesInputs(orderVal(1:80),:);
valImagesLabels = valImagesLabels(orderVal(1:80),:);

testImagesInputs= testImagesInputs(orderTest(1:80),:);
testImagesLabels = testImagesLabels(orderTest(1:80),:);
%%
% train
%%% Convert to 4-D Double
% Loop over filenames, inserting the image.
numTrainImages = numel(trainingImagesInputs);
array4dTrainImagesCal = uint8(zeros(224, 224,3, numTrainImages));
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
  array4dTrainImagesCal(:,:,:,slice) = imread(trainingImagesInputs{slice,1});
end

% val
numValImages = numel(valImagesInputs);
array4dValImagesCal = uint8(zeros(224, 224,3, numValImages));
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
  array4dValImagesCal(:,:,:,slice) = imread(valImagesInputs{slice,1});
end
% test
numTestImages = numel(testImagesInputs);
array4dTestImagesCal = uint8(zeros(224, 224,3, numTestImages));
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
  array4dTestImagesCal(:,:,:,slice) = imread(testImagesInputs{slice,1});
end

%%
save(fullfile(pwd,'Cal','18','array4dTrainImagesCal'),'array4dTrainImagesCal', '-v7.3');
save(fullfile(pwd,'Cal','18','array4dValImagesCal'),'array4dValImagesCal', '-v7.3');
save(fullfile(pwd,'Cal','18','array4dTestImagesCal'),'array4dTestImagesCal', '-v7.3');

save(fullfile(pwd,'Cal','18','trainingImagesLabels'),'trainingImagesLabels', '-v7.3');
save(fullfile(pwd,'Cal','18','valImagesLabels'),'valImagesLabels', '-v7.3');
save(fullfile(pwd,'Cal','18','testImagesLabels'),'testImagesLabels', '-v7.3');

%  save(fullfile(pwd,'Cal/18/orderTrain'),'orderTrain', '-v7.3');
%  save(fullfile(pwd,'Cal/18/orderTest'),'orderTest', '-v7.3');
%  save(fullfile(pwd,'Cal/18/orderVal'),'orderVal', '-v7.3');
%%
matTrainingEEGFeaturesLabels =averageEncoderOutputTrain;
%%
function imagesReziseSelected
mainFolder = fullfile(pwd,'Selected18');
foldersPaths = dir(fullfile(mainFolder));
for i = 3 : size(foldersPaths,1)
    subFolder=foldersPaths(i).name;
    filesPaths = dir(fullfile(foldersPaths(i).folder, subFolder,'*.jpg'));
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
