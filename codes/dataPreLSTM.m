clc
clear

w=1;
numClasses=40;
nunfiles = 0;
tarinCounter=1;
testCounter=1;
valCounter=1;

sizeClassSplit = 0;
acSizeClassSplit = 0;

trainInputsNames = cell(9191,1);
trainInputs = cell(9191,1);
trainLabels = cell (9191,1);

testInputsNames = cell(1127,1);
testInputs = cell (1127,1);
testLabels = cell(1127,1);

valInputsNames = cell(1148,1);
valInputs = cell (1148,1);
valLabels = cell(1148,1);

dataSet = fullfile(pwd,'Dataset');
all6Subject = dir(fullfile(dataSet));

%%
for i = 3 : size(all6Subject,1) % loop of subject 6
    w=1;
    subject=all6Subject(i).name; %1 2 3 4 5 6
    subjectFiles = dir(fullfile(all6Subject(i).folder, subject,'*.mat'));
    nunfiles = nunfiles + length(subjectFiles);
    for c = 1 : numClasses % loop for classes 40 *50 = 2000
        fileName = subjectFiles(w).name;
        fileNameSplit= strsplit(fileName,'_');
        classname {c} = fileNameSplit{1};
        classSplit = dir(fullfile( all6Subject(i).folder,subject ,strcat('*', classname{c},'*.mat')));
        sizeClassSplit = length(classSplit); % Stop may sizeClassSplit =38 as in n13054560
        acSizeClassSplit = acSizeClassSplit + sizeClassSplit;
        if sizeClassSplit > 40
            for j =1 : 40
                trainInputsNames{tarinCounter,1} = classSplit(j).name;
                trainInputs{tarinCounter,1} = importdata(fullfile(all6Subject(i).folder,subject ,trainInputsNames{tarinCounter,1}))';
                trainInputs{tarinCounter,1} = trainInputs{tarinCounter,1}(:,41:480);
                temp= strsplit(classname{c},'n');
                trainLabels{tarinCounter,1} = temp {1,2};
                tarinCounter = tarinCounter+1;
            end
            x=round(((sizeClassSplit-40)/2));
            for k=1 : x %val
                valInputsNames {valCounter,1} =  classSplit(k+40).name;
                valInputs{valCounter,1} = importdata(fullfile(all6Subject(i).folder,subject ,valInputsNames{valCounter,1}))';
                valInputs{valCounter,1} = valInputs{valCounter,1}(:,41:480);
                temp = strsplit(classname{c},'n');
                valLabels {valCounter,1} = temp {1,2};
                valCounter = valCounter+1;
            end
            
            for m=1 : floor(((sizeClassSplit-40)/2))
                testInputsNames {testCounter,1} =  classSplit(m+40+x).name;
                testInputs{testCounter,1} = importdata(fullfile(all6Subject(i).folder,subject ,testInputsNames{testCounter,1}))';
                testInputs{testCounter,1} = testInputs{testCounter,1}(:,41:480);
                temp = strsplit(classname{c},'n');
                testLabels {testCounter,1} = temp {1,2};
                testCounter = testCounter+1;
            end
        else                                % Stop may sizeClassSplit =38 as in n13054560
            for j =1 : 31
                trainInputsNames{tarinCounter,1} = classSplit(j).name;
                trainInputs{tarinCounter,1} = importdata(fullfile(all6Subject(i).folder,subject ,trainInputsNames{tarinCounter,1}))';
                trainInputs{tarinCounter,1} = trainInputs{tarinCounter,1}(:,41:480);
                temp= strsplit(classname{c},'n');
                trainLabels{tarinCounter,1} = temp {1,2};
                tarinCounter = tarinCounter+1;
            end
            x=floor(((sizeClassSplit-31)/2));
            for k=1 : x
                valInputsNames {valCounter,1} =  classSplit(k+31).name;
                valInputs{valCounter,1} = importdata(fullfile(all6Subject(i).folder,subject ,valInputsNames{valCounter,1}))';
                valInputs{valCounter,1} = valInputs{valCounter,1}(:,41:480);
                temp = strsplit(classname{c},'n');
                valLabels {valCounter,1} = temp {1,2};
                valCounter = valCounter+1;
            end
            for m=1 : round(((sizeClassSplit-31)/2))
                testInputsNames {testCounter,1} =  classSplit(m+31+x).name;
                testInputs{testCounter,1} = importdata(fullfile(all6Subject(i).folder,subject ,testInputsNames{testCounter,1}))';
                testInputs{testCounter,1} = testInputs{testCounter,1}(:,41:480);
                temp = strsplit(classname{c},'n');
                testLabels {testCounter,1} = temp {1,2};
                testCounter = testCounter+1;
            end
        end
        w=w+sizeClassSplit;
        if length(subjectFiles) == 1497 && c == 30
            break;
        end
    end
end

%%

[C,ia,ib] = intersect(trainInputsNames,testInputsNames)
[C,ia,ib] = intersect(valInputsNames,testInputsNames)


%%
load(fullfile(pwd,'Data/orderTrain'));
load(fullfile(pwd,'Data/orderTest'));
load(fullfile(pwd,'Data/orderVal'));
% orderTrain = randperm(9191)';
% orderTest = randperm (1127)';
% orderVal = randperm (1148)';

trainInputs= trainInputs(orderTrain(1:9191),:);
trainInputsNames = trainInputsNames(orderTrain(1:9191),:);
trainLabels = trainLabels(orderTrain(1:9191),:);

valInputs = valInputs(orderVal(1:1148),:);
valInputsNames = valInputsNames(orderVal(1:1148),:);
valLabels = valLabels (orderVal(1:1148),:);

testInputs = testInputs(orderTest(1:1127),:);
testInputsNames = testInputsNames(orderTest(1:1127),:);
testLabels = testLabels (orderTest(1:1127),:);

%%
trainLabels = categorical(trainLabels);
testLabels = categorical (testLabels);
%%
%addpath('F:\Future\MatWork\Data');
save(fullfile(pwd,'Data/trainInputs') ,'trainInputs', '-v7.3'); % traning inputs 80%
save(fullfile(pwd,'Data/trainLabels'),'trainLabels', '-v7.3'); % train lables
save(fullfile(pwd,'Data/testInputs'),'testInputs', '-v7.3'); % test inputs
save(fullfile(pwd,'Data/testLabels'),'testLabels', '-v7.3'); % test labels
save(fullfile(pwd,'Data/valInputs'),'valInputs', '-v7.3'); % test inputs
save(fullfile(pwd,'Data/valLabels'),'valLabels', '-v7.3'); % test labels
%%
save(fullfile(pwd,'Data/trainInputsNames'),'trainInputsNames', '-v7.3');
save(fullfile(pwd,'Data/testInputsNames'),'testInputsNames', '-v7.3');
save(fullfile(pwd,'Data/valInputsNames'),'valInputsNames', '-v7.3');
%%
% save(fullfile(pwd,'Data/orderTrain'),'orderTrain', '-v7.3');
% save(fullfile(pwd,'Data/orderTest'),'orderTest', '-v7.3');
% save(fullfile(pwd,'Data/orderVal'),'orderVal', '-v7.3');
