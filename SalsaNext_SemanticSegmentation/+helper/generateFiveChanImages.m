function generateFiveChanImages(dataFolder,imagesFolder)

if ~exist(imagesFolder,'dir')
    mkdir(imageDataLocation);
end

rangeFolder = dir(fullfile(dataFolder,'*.png'));
numFiles = size(rangeFolder,1);

for ii = 1:numFiles
    % Load images and calculate xyz
    Img = double(imread(fullfile(rangeFolder(ii).folder, rangeFolder(ii).name)));
    Range = Img(:,:,3);
    Intensity = Img(:,:,1);

    % Set outliers to NaN
    lowNoise = 1; 
    highNoise =254;
    OutLog = Range < lowNoise | Range > highNoise;
    Range(OutLog) = NaN;
    Intensity(OutLog) = NaN;
    XYZ = helper.calculateXYZ(Range);

    % Image are of 5-channels, namely x,y,z,intensity and range.
    Img5ch = zeros([size(Img,[1,2]), 5]);
    Img5ch(:,:,1:3) = XYZ;
    Img5ch(:,:,4) = Intensity;
    Img5ch(:,:,5) = Range;
    
    
    % Store images and labels as .mat and .png files respectively.
    imfile = fullfile(imagesFolder,sprintf('%04d.mat',ii));
    save(imfile,'Img5ch');   
end
end