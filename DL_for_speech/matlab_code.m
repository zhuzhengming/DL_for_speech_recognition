clear all;
close all force;

% define the random number seed for repeatable results
rng(1,'twister');
timePoolSize = 12;

%% Load Speech Data 

% create an image data store from the raw images 
imdsTrain = imageDatastore('C:\Users\Administrator\Desktop\DL_for_speech\speechImageData\speechImageData\TrainData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

% create an image validation data store from the validation images 
imdsVal = imageDatastore('C:\Users\Administrator\Desktop\DL_for_speech\speechImageData\speechImageData\ValData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

%%

% your code here...
% build network

% grid search:
% conv layers 2 + fliters 8  conv layers 3 + fliters 8 
% conv layers 2 + fliters 16  conv layers 3 + fliters 16 

% The number of convolution layers: 2
% The number of filters of first convolution layer:8
% Net= [
%     imageInputLayer([98 50 1],'Name','Image input')
%     convolution2dLayer([3 3],8,'Padding','same','Name','Conv1')
%     batchNormalizationLayer('Name','batchnorm_1')
%     reluLayer('Name','relu_1')
%     maxPooling2dLayer(2,'Stride',2,'Padding','same','Name','maxpool1')
% 
%     convolution2dLayer([3 3],16,'Padding','same','Stride', 2 ,'Name','Conv2')
%     batchNormalizationLayer('Name','batchnorm_2')
%     reluLayer('Name','relu_2')
%     maxPooling2dLayer([timePoolSize 1],'Name','maxpool2')
% 
%     dropoutLayer(0.15,'Name','Dropout')
%     fullyConnectedLayer(12,'Name','fc')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','output')
%     ];

% The number of convolution layers: 2
% The number of filters of firsr convolution layer:16
% Net= [
%     imageInputLayer([98 50 1],'Name','Image input')
%     convolution2dLayer([3 3],16,'Padding','same','Name','Conv1')
%     batchNormalizationLayer('Name','batchnorm_1')
%     reluLayer('Name','relu_1')
%     maxPooling2dLayer(2,'Stride',2,'Padding','same','Name','maxpool1')
% 
%     convolution2dLayer([3 3],32,'Padding','same','Stride', 2 ,'Name','Conv2')
%     batchNormalizationLayer('Name','batchnorm_2')
%     reluLayer('Name','relu_2')
%     maxPooling2dLayer([timePoolSize 1],'Name','maxpool2')
% 
%     dropoutLayer(0.15,'Name','Dropout')
%     fullyConnectedLayer(12,'Name','fc')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','output')
%     ];

% The number of convolution layers: 3
% The number of filters of first convolution layer:8
% Net= [
%     imageInputLayer([98 50 1],'Name','Image input')
%     convolution2dLayer([3 3],8,'Padding','same','Name','Conv1')
%     batchNormalizationLayer('Name','batchnorm_1')
%     reluLayer('Name','relu_1')
%     maxPooling2dLayer(2,'Stride',2,'Padding','same','Name','maxpool1')
% 
%     convolution2dLayer([3 3],16,'Padding','same','Stride', 2 ,'Name','Conv2')
%     batchNormalizationLayer('Name','batchnorm_2')
%     reluLayer('Name','relu_2')
%     convolution2dLayer([3 3],32,'Padding','same','Stride', 2 ,'Name','Conv3')
%     batchNormalizationLayer('Name','batchnorm_3')
%     reluLayer('Name','relu_3')
%     maxPooling2dLayer([timePoolSize 1],'Name','maxpool2')
% 
%     dropoutLayer(0.15,'Name','Dropout')
%     fullyConnectedLayer(12,'Name','fc')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','output')
%     ];

% The number of convolution layers: 3
% The number of filters of first convolution layer:16
% Net= [
%     imageInputLayer([98 50 1],'Name','Image input')
%     convolution2dLayer([3 3],16,'Padding','same','Name','Conv1')
%     batchNormalizationLayer('Name','batchnorm_1')
%     reluLayer('Name','relu_1')
%     maxPooling2dLayer(2,'Stride',2,'Padding','same','Name','maxpool1')
% 
%     convolution2dLayer([3 3],32,'Padding','same','Stride', 2 ,'Name','Conv2')
%     batchNormalizationLayer('Name','batchnorm_2')
%     reluLayer('Name','relu_2')
%     convolution2dLayer([3 3],64,'Padding','same','Stride', 2 ,'Name','Conv3')
%     batchNormalizationLayer('Name','batchnorm_3')
%     reluLayer('Name','relu_3')
%     maxPooling2dLayer([timePoolSize 1],'Name','maxpool2')
% 
%     dropoutLayer(0.15,'Name','Dropout')
%     fullyConnectedLayer(12,'Name','fc')
%     softmaxLayer('Name','softmax')
%     classificationLayer('Name','output')
%     ];

%  Reduce computational complexity by depthwise separable convolutions and
%  1x1 convolutions.
Net= [
    imageInputLayer([98 50 1],'Name','Image input')
    % depthwise separable convolutions
    convolution2dLayer([3 3],1,'Padding','same','Name','Conv1')
    convolution2dLayer([1 1],16,'Padding','same','Name','Conv2')
    batchNormalizationLayer('Name','batchnorm_1')
    reluLayer('Name','relu_1')
    maxPooling2dLayer(2,'Stride',2,'Padding','same','Name','maxpool1')

    % depthwise separable convolutions
    convolution2dLayer([3 3],1,'Padding','same','Stride', 2 ,'Name','Conv3')
    convolution2dLayer([1 1],32,'Padding','same','Name','Conv4')
    batchNormalizationLayer('Name','batchnorm_2')
    reluLayer('Name','relu_2')
    convolution2dLayer([3 3],64,'Padding','same','Stride', 2 ,'Name','Conv5')
    batchNormalizationLayer('Name','batchnorm_3')
    reluLayer('Name','relu_3')
    maxPooling2dLayer([timePoolSize 1],'Name','maxpool2')

    dropoutLayer(0.15,'Name','Dropout')
    fullyConnectedLayer(12,'Name','fc')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
    ];
    
%Visualize the network
lgraph = layerGraph(Net);
analyzeNetwork(lgraph);

% Configuration 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.025, ...  
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'MaxEpochs',150, ...               
    'Shuffle','every-epoch', ...
    'ValidationData',imdsVal, ...
    'ValidationFrequency',50, ...
    'Verbose',true, ...
    'Plots','training-progress'); 

net = trainNetwork(imdsTrain,Net,options);

% evaluation
YPred = classify(net,imdsVal);
YValidation = imdsVal.Labels;
accuracy = sum(YPred == YValidation)/numel(YValidation);
fprintf('Accuracy: %.2f%%\n',accuracy*100);

figure
cm = confusionchart(YValidation,YPred);
cm.ColumnSummary ="column-normalized";
cm.RowSummary = "row-normalized";
title('Confusion Matirx for Validation Data');
% cm.Normalization = 'absolute'; 




