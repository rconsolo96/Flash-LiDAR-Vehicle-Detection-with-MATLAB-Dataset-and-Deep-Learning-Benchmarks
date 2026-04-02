%% Train SalsaNext Model for Vehicle Detection in Flash Lidar Data

%% Load Pretrained Model
net = load('SalsaNext_Randomized.mat').net; 
% net = load('SalsaNext_Pretrained.mat').net;

%% Load Datastores
labelsFolder = "..\..\Masks"; %Switch to directory where the data was downloaded
imagesFolder = "..\..\Images_5Ch"; %Switch to directory where the data was downloaded

imds = imageDatastore(imagesFolder, ...
    'FileExtensions', '.mat', ...
    'ReadFcn', @helper.imageMatReader, ...
    'IncludeSubfolders',true);
 

classNames = ["Undefined"
              "Background"
              "ATV"
              "JeepGreen"
              "PickupWhite"
              "SUVBlack"
              "VanWhite"
              "Plane"
              "SedanBlack"
              "SportsCarYellow"];

numClasses = numel(classNames);

% Specify label IDs from 1 to the number of classes.
labelIDs = 0 : numClasses-1;

pxds = pixelLabelDatastore(labelsFolder, classNames, labelIDs, "IncludeSubfolders",true);

%% Prepare Training, Validation, and Test Sets
% Use the partitionLidarData helper function to split the data into
% training, images, respectively.

[imdsTrain, imdsVal, imdsTest, pxdsTrain, pxdsVal, pxdsTest] = helper.partitionLidarData(imds, pxds);

dsTrain = combine(imdsTrain,pxdsTrain);
dsVal = combine(imdsVal,pxdsVal);


%% Configure Pretrained Network

% Changing output size to required number of classes.
inputSize = [128, 128, 5];
net = replaceLayer(net, 'Input_input.1', imageInputLayer(inputSize, 'Name', 'Input_input.1', 'Normalization', 'none'));
net = replaceLayer(net, 'Conv_191', convolution2dLayer([1,1], numClasses, 'Name', 'Conv_191'));

%% % Define training options. 
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',1,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.9, ...
    'InitialLearnRate',1e-3, ...
    'L2Regularization',0.005, ...
    'ValidationData',dsVal,...
    'MaxEpochs',30, ...  
    'MiniBatchSize',40, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2,...
    'Plots','training-progress',...
    'ValidationPatience', 4, ...
    'ValidationFrequency', 500);


%% Train the network
doTraining = true;

if doTraining
    weights = [0.02 0.02 0.12 0.12 0.12 0.12 0.12 0.12 0.12 0.12];
    [trainedNet,info] = trainnet(dsTrain,net, @(Y,T)crossentropy(Y,T,weights,WeightsFormat="UC"),options);
    save('SalsaNext_Trained_W.mat', 'trainedNet');
else
    load("SalsaNext_Trained.mat","trainedNet");
end

%% Run model on Test data
testResults = semanticseg(imdsTest, trainedNet, "Classes", classNames);

%%  Evaluate metrics
% testResultsT = transform(testResults,@(x) helper.assignCategories(x, classNames));
% metrics = evaluateSemanticSegmentation(testResultsT,pxdsTest);
metrics = evaluateSemanticSegmentation(testResults,pxdsTest);

%% Visualize metrics
metrics.ClassMetrics
metrics.ConfusionMatrix

%% Test data and visualize results
figure('Position', [50 50 1800 900])

while hasdata(imdsTest)
    testImg = read(imdsTest);
    testLabels = read(pxdsTest);
    %     testImg = read(imds);
    % testLabels = read(pxds);
    predictedResult = semanticseg(testImg,trainedNet); %,"Classes",classNames
    
    
    tiledlayout(1, 2);
    % Display Ground Truth 
    nexttile
    helper.displayLidarOverlayImage(testImg, testLabels{1}, classNames);
    title('Semantic Segmentation Ground Truth');
    % Display Predicted Output 
    nexttile
    helper.displayLidarOverlayImage(testImg, predictedResult, classNames);
    title('Semantic Segmentation Result');
    drawnow
    pause
end


% % Display in point cloud format.
% cmap = helper.lidarColorMap();
% colormap = cmap(single(predictedResult),:);
% ptCloudMod = pointCloud(reshape(I(:,:,1:3),[],3),"Color",colormap);
% figure
% ax = pcshow(ptCloudMod);
% zoom(ax,3);

% Copyright 2026 The MathWorks, Inc.