clear all;

% define the random number seed for repeatable results
rng(1,'twister');

%% Load Speech Data 

% create an image data store from the raw images 
imdsTrain = imageDatastore('speechImageData\TrainData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

% create an image validation data store from the validation images 
imdsVal = imageDatastore('speechImageData\ValData',...
"IncludeSubfolders",true,"LabelSource","foldernames")

%%

% your code here...

