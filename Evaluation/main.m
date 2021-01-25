clear all; close all; clc;

%set your dataset path and saliency map result path.
dataset = 'SIP';
salPath = 'C:\Users\user02\Documents\GitHub\EfficientSOD\SIPTestResults/';
gtPath = 'D:\My Research\Datasets\Saliency Detection\SIP\Test\Labels\';

% dataset = 'NLPR';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\NLPR Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\NLPR\Test\Labels\';

% dataset = 'NJU2K';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\NJU2K Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\NJU2K\Test\Labels\';

% Testing DUTRGBD
% dataset = 'DUTRGB-D';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\DUTRGB-D Test Data Results\Temp/';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\DUT-RGBD\Test\Labels/';

% Testing Chokepiont data
% dataset = 'Choke';
% salPath = 'D:\Research Group\Research circle\Dr. Saeed Anwar\VS via Saliency\Experimental results\CrossData\EfficientNet_binary_crossentropy_SIPData\';
% gtPath = 'D:\My Research\Datasets\Saliency Detection\SOD_Dataset\S-SOD\Labels\';

%obtain the total number of image (ground-truth)
imgFiles = dir(gtPath);
imgNUM = length(imgFiles)-2;

%evaluation score initilization.
Smeasure=zeros(1,imgNUM);
Emeasure=zeros(1,imgNUM);
Fmeasure=zeros(1,imgNUM);
MAE=zeros(1,imgNUM);
F_wm=zeros(1,imgNUM);

tic;
for i = 1:imgNUM
    
    fprintf('Evaluating: %d/%d\n',i,imgNUM);
    
    name =  imgFiles(i+2).name;
    %name = name(:,3:10);
    
    %load gt
    gt = imread([gtPath name]);
    
    if numel(size(gt))>2
        gt = rgb2gray(gt);
    end
    if ~islogical(gt)
        gt = gt(:,:,1) > 128;
    end
    
    %load salency
    sal  = imread([salPath name]);
    
    if numel(size(sal))>2
        sal = rgb2gray(sal);
    end
    if ~islogical(sal)
        sal = sal(:,:,1) > 50;
    end
    
    %check size
    if size(sal, 1) ~= size(gt, 1) || size(sal, 2) ~= size(gt, 2)
        sal = imresize(sal,size(gt));
        imwrite(sal,[salPath name]);
        fprintf('Error occurs in the path: %s!!!\n', [salPath name]);
       
    end
    
    sal = im2double(sal(:,:,1));
    
    %normalize sal to [0, 1]
    sal = reshape(mapminmax(sal(:)',0,1),size(sal));
    
    Smeasure(i) = StructureMeasure(sal,logical(gt));
    temp = Fmeasure_calu(sal,double(gt),size(gt)); % Using the 2 times of average of sal map as the threshold.
    Fmeasure(i) = temp(3);
   
    MAE(i) = mean2(abs(double(logical(gt)) - sal));
    F_wm(i) = WFb(sal, logical(gt));
    
    %You can change the method of binarization method. As an example, here just use adaptive threshold.
    threshold =  2* mean(sal(:)) ;
    if ( threshold > 1 )
        threshold = 1;
    end
    Bi_sal = zeros(size(sal));
    Bi_sal(sal>threshold)=1;
    Emeasure(i) = Enhancedmeasure(Bi_sal,gt);
    
end

toc;

Sm = mean2(Smeasure);
Fm = mean2(Fmeasure);
Em = mean2(Emeasure);
mae = mean2(MAE);
F_wm = mean2(F_wm);

fprintf('(%s Dataset)Emeasure: %.3f; Smeasure %.3f; weighted_F: %.3f; Fmeasure %.3f; MAE: %.3f.\n',dataset,Em, Sm, F_wm, Fm, mae);


