% this version work on Caltech-101 to test the generalization ability of our model in transfere learning
% indirict mapping from the trainInameNet-->TrainCal-->TestCal

% accTest13 = 0.966 --> Mdl13
% accTest15 = 0.977 --> Mdl15

clc
clear
gpuDevice(1);
DataPreGoogleNetCal;
load(fullfile(pwd,'DataGoogleNet','array4dTrainImages'));

%%
% Display some sample images.
numTrainImages = size(array4dTrainImages,4);
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
trainingFeatures = activations(netGoogle,array4dTrainImages,layer);
%trainingFeatures = activations(netGoogle,imread(trainingImagesInputs{1,1}),layer);
trainingFeaturesCal = activations(netGoogle,array4dTrainImagesCal,layer);
testFeaturesCal = activations(netGoogle,array4dTestImagesCal,layer);

trainingLabels = matTrainingEEGFeaturesLabels;
%testLabels = matTestEEGFeaturesLabels;

%%
trainingFeatures = reshape(trainingFeatures,[1000 numTrainImages])';
trainingFeaturesCal = reshape(trainingFeaturesCal,[1000 numTrainImagesCal])';
testFeaturesCal = reshape(testFeaturesCal, [1000 numTestImagesCal])';

predictedLabelsTrainCal = zeros(numTrainImagesCal,128);
for i = 1: numTrainImagesCal
 [predictedLabelsTrainCal(i,:),IDX,D] = KNNregressor(trainingFeatures,trainingFeaturesCal(i,:),trainingLabels);
end

predictedLabelsTestCal = zeros(numTestImagesCal,128);
for i = 1: numTestImagesCal
 [predictedLabelsTestCal(i,:),IDX,D] = KNNregressor(trainingFeaturesCal,testFeaturesCal(i,:),predictedLabelsTrainCal);
end

% predictionError = testLabels - predictedLabels;
% rmse= sqrt(meansqr(predictionError));
%%
% save(fullfile(pwd,'Cal','predictedLabelsTestCal'),'predictedLabelsTestCal', '-v7.3');
% save(fullfile(pwd,'Cal','predictedLabelsTrainCal'),'predictedLabelsTrainCal', '-v7.3');
%% SoftMax and SVM
%SoftMax
getClassifierOutput3;

% SVM
load(fullfile(pwd,'SVM','Mdl'));
load(fullfile(pwd,'SVM','Mdl2'));
load(fullfile(pwd,'SVM','Mdl3'));
load(fullfile(pwd,'Cal/testImagesLabels'));
load(fullfile(pwd,'Cal/trainingImagesLabels'));
testImagesLabels=categorical(testImagesLabels);
trainingImagesLabels=categorical(trainingImagesLabels);

%without RICA
classifierOutputs0 = predict(Mdl, predictedLabelsTestCal);
accTest0 = mean(classifierOutputs0 == testImagesLabels);

%with RICA
TestX = transform(Mdl2,predictedLabelsTestCal);
classifierOutputs1 = predict(Mdl3, TestX);
accTest1 = mean(classifierOutputs1 == testImagesLabels);

% SVM Retrain without RICA
t = templateSVM('Standardize',1,'SaveSupportVectors',true);
Mdl13 = fitcecoc(predictedLabelsTrainCal,trainingImagesLabels,'Learners',t,'FitPosterior',1,'Verbose',2);
predictedLabels13 = predict(Mdl13, predictedLabelsTestCal);
accTest13 = mean(predictedLabels13 == testImagesLabels);

% SVM Retrain with RICA
q = 70;
Mdl14 = rica(predictedLabelsTrainCal,q,'IterationLimit',400,'Standardize',true);
NewX = transform(Mdl14,predictedLabelsTrainCal);
TestX = transform(Mdl14,predictedLabelsTestCal);

t = templateSVM('Standardize',1,'SaveSupportVectors',true);
Mdl15 = fitcecoc(NewX,trainingImagesLabels,'Learners',t,'FitPosterior',1,'Verbose',2);
predictedLabels15 = predict(Mdl15, TestX);
accTest15 = mean(predictedLabels15 == testImagesLabels);

%
% save(fullfile(pwd,'SVM','Mdl13'),'Mdl13', '-v7.3');
% save(fullfile(pwd,'SVM','Mdl15'),'Mdl15', '-v7.3');


%%
function [predictedLabels,IDX,D]=KNNregressor(trainingFeatures,testFeatures,trainingLabels)

[IDX,D] = knnsearch(trainingFeatures,testFeatures,'k',5);
predictedLabels=mean(trainingLabels(IDX,:));
end
