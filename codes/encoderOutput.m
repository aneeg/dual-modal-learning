% o/p directly form debugging the LSTM classifiy 
%% Return the EEG feature representaion
clc
clear

load(fullfile(pwd,'Data/trainInputs.mat'));
load(fullfile(pwd,'Data/testInputs.mat'));
load(fullfile(pwd,'Data/valInputs.mat'));
%load(fullfile(pwd,'netFull4'));
load(fullfile(pwd,'Data/trainInputsNames')); % with the same perm as train
load(fullfile(pwd,'Data/testInputsNames'));  % with the same perm as test
load(fullfile(pwd,'Data/valInputsNames')); 
%%
ReLULayerOutputTrain1=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain1.mat'));
ReLULayerOutputTrain2=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain2.mat'));
ReLULayerOutputTrain3=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain3.mat'));
ReLULayerOutputTrain4=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain4.mat'));
ReLULayerOutputTrain5=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain5.mat'));
ReLULayerOutputTrain6=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain6.mat'));
ReLULayerOutputTrain7=load(fullfile(pwd,'EncoderOutput','Train','ReLULayerOutputTrain7.mat'));

ReLULayerOutputVal=load(fullfile(pwd,'EncoderOutput','Val','ReLULayerOutputVal.mat'));


ReLULayerOutputTest=load(fullfile(pwd,'EncoderOutput','Test','ReLULayerOutputTest.mat'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GET LSTM OUTPUT of the pretrained LSTM
OutputEncodedFeaturesTrain = [ReLULayerOutputTrain1.output,ReLULayerOutputTrain2.output, ...
    ReLULayerOutputTrain3.output, ReLULayerOutputTrain4.output, ReLULayerOutputTrain5.output, ...
    ReLULayerOutputTrain6.output, ReLULayerOutputTrain7.output];
OutputEncodedFeaturesTrain = gather(OutputEncodedFeaturesTrain)';

OutputEncodedFeaturesVal = ReLULayerOutputVal.output;
OutputEncodedFeaturesVal = gather(OutputEncodedFeaturesVal)';

OutputEncodedFeaturesTest = ReLULayerOutputTest.output;
OutputEncodedFeaturesTest = gather(OutputEncodedFeaturesTest)';


%%
save(fullfile(pwd,'Data/OutputEncodedFeaturesTrain'),'OutputEncodedFeaturesTrain', '-v7.3'); % activations lstm ->FC_1 -> ReLU
save(fullfile(pwd,'Data/OutputEncodedFeaturesTest'),'OutputEncodedFeaturesTest','-v7.3');
save(fullfile(pwd,'Data/OutputEncodedFeaturesVal'),'OutputEncodedFeaturesVal','-v7.3');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%Take the average -- for each 1 image we have 6 eeg fearure vector take the avrage.
% you will have about 1996 eeg fearure vector, one for each image

%% encoder


%% Name and index for train and test
[NameAndGroupIndexOfEachImageTrain, NameAndGroupIndexOfEachImageTest, NameAndGroupIndexOfEachImageVal,...
    catNameUniqueTrain, catNameUniqueTest, catNameUniqueVal] = groupIndexOfEachImage2...
    (trainInputsNames,testInputsNames,valInputsNames);
%% Train
numEncoderTrain = size(OutputEncodedFeaturesTrain ,1); % encoder counter 10339
numGroupTrain = size(NameAndGroupIndexOfEachImageTrain ,1); % 1800

%averageEncoderOutputTrain = cell (numGroupTrain,1);

averageEncoderOutputTrain = zeros (numGroupTrain,128);


for CounterGroup = 1 : numGroupTrain   % 1800
    idx = NameAndGroupIndexOfEachImageTrain{CounterGroup,2};
   
    averageEncoderOutputTrain(CounterGroup , :)= mean(OutputEncodedFeaturesTrain(idx,:));
    
end


%% Test
numEncoderTest = size(OutputEncodedFeaturesTest ,1); % encoder counter
numGroupTest = size(NameAndGroupIndexOfEachImageTest ,1);


averageEncoderOutputTest = zeros (numGroupTest,128);

for CounterGroup = 1 : numGroupTest   %
    idx = NameAndGroupIndexOfEachImageTest{CounterGroup,2};
    
    averageEncoderOutputTest(CounterGroup , :) = mean(OutputEncodedFeaturesTest(idx,:));
end
%% val
numEncoderVal = size(OutputEncodedFeaturesVal ,1); % encoder counter
numGroupVal = size(NameAndGroupIndexOfEachImageVal ,1);


averageEncoderOutputVal = zeros (numGroupVal,128);

for CounterGroup = 1 : numGroupVal   %
    idx = NameAndGroupIndexOfEachImageVal{CounterGroup,2};
    
    averageEncoderOutputVal(CounterGroup , :) = mean(OutputEncodedFeaturesVal(idx,:));
end
%%
save(fullfile(pwd,'Data/averageEncoderOutputTrain'),'averageEncoderOutputTrain', '-v7.3');
save(fullfile(pwd,'Data/averageEncoderOutputTest'),'averageEncoderOutputTest', '-v7.3');
save(fullfile(pwd,'Data/averageEncoderOutputVal'),'averageEncoderOutputVal', '-v7.3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
save(fullfile(pwd,'Data/catNameUniqueTrain'),'catNameUniqueTrain');
save(fullfile(pwd,'Data/catNameUniqueTest'),'catNameUniqueTest');
save(fullfile(pwd,'Data/catNameUniqueVal'),'catNameUniqueVal');

%%
function [NameAndGroupIndexOfEachImageTrain, NameAndGroupIndexOfEachImageTest, NameAndGroupIndexOfEachImageVal,...
    catNameUniqueTrain, catNameUniqueTest, catNameUniqueVal] = groupIndexOfEachImage2...
    (trainInputsNames,testInputsNames,valInputsNames)
%

%load('/data/ahmedfares/MatWork/Data/trainInputsNames');
catNameUniqueTrain = unique (trainInputsNames,'stable'); % note with the sam permutation as train set
numCatNameUniqueTrain = size(catNameUniqueTrain, 1); %1800
indexNameImageTrain=cell(size(catNameUniqueTrain,1),2); %1800*2
numFilesTrain = size(trainInputsNames,1);  %

for i = 1: numCatNameUniqueTrain  %1800
    indexNameImageTrain{i,1}= catNameUniqueTrain{i,1};
    for fileConter = 1 : numFilesTrain %
        if strcmp (catNameUniqueTrain{i,1},trainInputsNames{fileConter,1})
            indexNameImageTrain{i,2} = [ indexNameImageTrain{i,2}  fileConter ];
        end
    end
end
NameAndGroupIndexOfEachImageTrain = indexNameImageTrain;
%%

%load('/data/ahmedfares/MatWork/Data/testInputsNames');

catNameUniqueTest = unique (testInputsNames,'stable');
numCatNameUniqueTest = size(catNameUniqueTest, 1);
indexNameImageTest=cell(size(catNameUniqueTest,1),2);
numFilesTest = size(testInputsNames,1);

for i = 1: numCatNameUniqueTest
    indexNameImageTest{i,1}= catNameUniqueTest{i,1};
    for fileConter = 1 : numFilesTest
        if strcmp (catNameUniqueTest{i,1},testInputsNames{fileConter,1})
            indexNameImageTest{i,2} = [ indexNameImageTest{i,2}  fileConter ];
        end
    end
end
NameAndGroupIndexOfEachImageTest = indexNameImageTest;
%%
catNameUniqueVal = unique (valInputsNames,'stable');
numCatNameUniqueVal = size(catNameUniqueVal, 1);
indexNameImageVal=cell(size(catNameUniqueVal,1),2);
numFilesVal = size(valInputsNames,1);

for i = 1: numCatNameUniqueVal
    indexNameImageVal{i,1}= catNameUniqueVal{i,1};
    for fileConter = 1 : numFilesVal
        if strcmp (catNameUniqueVal{i,1},valInputsNames{fileConter,1})
            indexNameImageVal{i,2} = [ indexNameImageVal{i,2}  fileConter ];
        end
    end
end
NameAndGroupIndexOfEachImageVal = indexNameImageVal;
end
