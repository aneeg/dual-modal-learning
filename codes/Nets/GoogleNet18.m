% GoogLeNet plus Cal  ---> SVM Original 
% accTest16 = 0.975000000000000
clc
clear
gpuDevice(1);
load(fullfile(pwd, 'Cal','18','array4dTrainImagesCal'));
load(fullfile(pwd, 'Cal','18','array4dTestImagesCal'));
load(fullfile(pwd, 'Cal','18','trainingImagesLabels'));
load(fullfile(pwd, 'Cal','18','testImagesLabels'));
testImagesLabels=categorical(testImagesLabels);
trainingImagesLabels=categorical(trainingImagesLabels);

%%
% Display some sample images.
numTrainImagesCal = size(array4dTrainImagesCal,4);
numTestImagesCal = size(array4dTestImagesCal,4);
% idx = randperm(numTrainImages,16);
% figure
% for i = 1:16
%     subplot(4,4,i)
%     I = imread(trainingImagesInputs{idx(i),1});
%     imshow(I)
% end
%%
% Load the pretrained GoogLeNet network. If the Neural Network Toolbox(TM)
% Model _for GoogLeNet Network_ is not installed, then the software
% provides a download link. GoogLeNet is trained on more than one million
% images and can classify images into 1000 object categories.
netGoogle = googlenet;

%%
% Extract the layer graph from the trained network and plot the layer
% graph.
lgraph = layerGraph(netGoogle);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)
%%
layer = 'loss3-classifier';
trainingFeaturesCal = activations(netGoogle,array4dTrainImagesCal,layer);
testFeaturesCal = activations(netGoogle,array4dTestImagesCal,layer);

%%
trainingFeaturesCal = reshape(trainingFeaturesCal,[1000 numTrainImagesCal])';
testFeaturesCal = reshape(testFeaturesCal, [1000 numTestImagesCal])';

%%
% save(fullfile(pwd,'Cal','predictedLabelsTestCal'),'predictedLabelsTestCal', '-v7.3');
% save(fullfile(pwd,'Cal','predictedLabelsTrainCal'),'predictedLabelsTrainCal', '-v7.3');

%% SVM

% SVM Retrain without RICA
t = templateSVM('Standardize',1,'SaveSupportVectors',true);
Mdl16 = fitcecoc(trainingFeaturesCal,trainingImagesLabels,'Learners',t,'FitPosterior',1,'Verbose',2);
predictedLabels16 = predict(Mdl16, testFeaturesCal);
accTest16 = mean(predictedLabels16 == testImagesLabels);
