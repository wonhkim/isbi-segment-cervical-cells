function [w,b] = train_nuclei_classifier(lambda, rectified_bbox_dim, hogcell_dim)
%
% @description
%   train the linear SVM classifier to detect nuclei
%
% @author
%   Wonhui Kim
%
% @contact
%   wonhui@stanford.edu
%


%% load train data

addpath('../../BSR/grouping/lib');
addpath('../Train45Test90')
addpath('../Train45Test90/gpb_train');

% load training images
object = load('isbi_train');
Images = object.ISBI_Train;
num_trains = length(Images);

% load ground-truth
GT = load('isbi_train_GT');
GT_nuclei = GT.train_Nuclei;

% load nuclei bounding box
object = load('isbi_train_bbox');
Bbox_nuclei = object.bbox_nuclei;


%% Feature extraction

if ~exist('HOG_TRAIN.mat','file')
    % Positive training images
    k = 1;
    clear HOG_POS
    % for each image
    for i = 1:num_trains

        num_cells = length(Bbox_nuclei{i});
        I = Images{i};

        % for each cell
        for j = 1:num_cells
            bbox = Bbox_nuclei{i}{j};
            w = bbox(3)-bbox(1);
            h = bbox(4)-bbox(2);

            pattern = [0 0 0 0;
                       -w/5 -h/5 w/5 h/5;
                       -w/6 -h/6 w/6 h/6];
%                        -w/4 -h/4 w/4 h/4;

            %%% mode: warp
            for p = 1:size(pattern,1)
                bbox_new = round(bbox + pattern(p,:));
                bbox_new(1) = max(bbox_new(1),1);
                bbox_new(2) = max(bbox_new(2),1);
                bbox_new(3) = min(bbox_new(3),512);
                bbox_new(4) = min(bbox_new(4),512);
                
                patch = I(bbox_new(2):bbox_new(4), bbox_new(1):bbox_new(3));
                if isempty(patch), continue; end
                patch = imresize(patch, [rectified_bbox_dim,rectified_bbox_dim]);
                hog = vl_hog(single(patch), hogcell_dim);
                HOG_POS{k} = hog;
                k = k+1;
            end
            
%             %%% mode: warp
%             patch = I(bbox(2):bbox(4),bbox(1):bbox(3));
%             patch = imresize(patch,[rectified_bbox_dim,rectified_bbox_dim]);
% 
%             hog = vl_hog(single(patch), hogcell_dim);
%             imhog = vl_hog('render', hog);
% %             figure(1); clf; imagesc(imhog); colormap gray
% 
%             HOG_POS{k} = hog;
%             k = k+1;
        end
    end

    % Negative training images %% hard negatives
    k = 1;
    clear HOG_NEG
    for i = 1:num_trains

        num_cells = length(Bbox_nuclei{i});
        I = Images{i};

        % for each cell
        for j = 1:num_cells
            bbox = Bbox_nuclei{i}{j};
            w = bbox(3)-bbox(1);
            h = bbox(4)-bbox(2);

            pattern = [w 0 w 0; -w 0 -w 0; 0 h 0 h; 0 -h 0 -h;
                w h w h; -w h -w h; w -h w -h; -w -h -w -h;
                -w -h w h]/2;

%             pattern = [pattern; w 0 w 0; -w 0 -w 0; 0 h 0 h; 0 -h 0 -h;
%                 w h w h; -w h -w h; w -h w -h; -w -h -w -h]*2/3;
            
            %%% mode: warp
            for p = 1:size(pattern,1)
                bbox_new = round(bbox + pattern(p,:));
                bbox_new(1) = max(bbox_new(1),1);
                bbox_new(2) = max(bbox_new(2),1);
                bbox_new(3) = min(bbox_new(3),512);
                bbox_new(4) = min(bbox_new(4),512);

                patch = I(bbox_new(2):bbox_new(4), bbox_new(1):bbox_new(3));
                if isempty(patch), continue; end
                patch = imresize(patch, [rectified_bbox_dim,rectified_bbox_dim]);
                hog = vl_hog(single(patch), hogcell_dim);
                HOG_NEG{k} = hog;
                k = k+1;
            end
        end
        for j = 1:num_cells
            bbox = [ceil(rand*512),ceil(rand*512),ceil(rand*512),ceil(rand*512)];
            w = 100;
            h = 100;
            pattern = [w 0 w 0; -w 0 -w 0; 0 h 0 h; 0 -h 0 -h;
                       w h w h; -w h -w h; w -h w -h; -w -h -w -h]/2;
            %%% mode: warp
            for p = 1:size(pattern,1)
                bbox_new = round(bbox + pattern(p,:));
                bbox_new(1) = max(bbox_new(1),1);
                bbox_new(2) = max(bbox_new(2),1);
                bbox_new(3) = min(bbox_new(3),512);
                bbox_new(4) = min(bbox_new(4),512);

                patch = I(bbox_new(2):bbox_new(4), bbox_new(1):bbox_new(3));
                if isempty(patch), continue; end
                patch = imresize(patch, [rectified_bbox_dim,rectified_bbox_dim]);
                hog = vl_hog(single(patch), hogcell_dim);
                HOG_NEG{k} = hog;
                k = k+1;
            end          
        end
        for j = 1:num_cells
            bbox = [ceil(rand*512),ceil(rand*512),ceil(rand*512),ceil(rand*512)];
            w = 200;
            h = 200;
            pattern = [w 0 w 0; -w 0 -w 0; 0 h 0 h; 0 -h 0 -h;
                       w h w h; -w h -w h; w -h w -h; -w -h -w -h]/2;
            %%% mode: warp
            for p = 1:size(pattern,1)
                bbox_new = round(bbox + pattern(p,:));
                bbox_new(1) = max(bbox_new(1),1);
                bbox_new(2) = max(bbox_new(2),1);
                bbox_new(3) = min(bbox_new(3),512);
                bbox_new(4) = min(bbox_new(4),512);

                patch = I(bbox_new(2):bbox_new(4), bbox_new(1):bbox_new(3));
                if isempty(patch), continue; end
                patch = imresize(patch, [rectified_bbox_dim,rectified_bbox_dim]);
                hog = vl_hog(single(patch), hogcell_dim);
                HOG_NEG{k} = hog;
                k = k+1;
            end          
        end
    end
    
    save('HOG_TRAIN.mat','HOG_POS','HOG_NEG','-v7.3')
else
    load HOG_TRAIN.mat
end

%% linear SVM
X = []; Y = [];
for i = 1:length(HOG_POS)
    X = [X HOG_POS{i}(:)];
    Y = [Y; 1];
end
for i = 1:length(HOG_NEG)
    X = [X HOG_NEG{i}(:)];
    Y = [Y; -1];
end

% lambda = 0.1;
[w,b] = vl_svmtrain(X,Y,lambda);



%%% mode: square
%         % crop bbox as a square
%         s = max(w,h);
%         if w == s && h ~= s
%             bbox(4) = bbox(4) + ceil((w-h)/2);
%             bbox(2) = bbox(2) - floor((w-h)/2);
%         elseif h==s && w ~= s
%             bbox(3) = bbox(3) + ceil((h-w)/2);
%             bbox(1) = bbox(1) - floor((h-w)/2);
%         else % w==h
%             % do nothing
%         end
%         
%         scale = rectified_bbox_dim / s;
%         patch = I(bbox(2):bbox(4),bbox(1):bbox(3));
%         patch = imresize(patch,[rectified_bbox_dim,rectified_bbox_dim]);