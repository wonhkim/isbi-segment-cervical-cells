function [bbox_nuclei, BW, labels, num_cells] = detect_nuclei(test_id)
%
% @description
%   nuclei detection
%
% @author
%   Wonhui Kim
%
% @contact
%   wonhui@stanford.edu
%


%% load test data

addpath('../../BSR/grouping/lib');
addpath('../Train45Test90/gpb_test'); % globalPb results are here!
addpath('../Train45Test90')

object = load('isbi_test90.mat');
D.test = object.ISBI_Test90;
D.test_gt = load('isbi_test90_GT.mat');


%% nuclei detection parameters
rectified_bbox_dim = 24;
hogcell_dim = 6;


%% keypoints detection by using MSER
% for test_id = 1:90
I = D.test{test_id}; % grayscale image

% extract keypoints
[r,f] = vl_mser(I,'MinDiversity',0.5,...
                  'MaxVariation',0.2,...
                  'Delta',5,...
                  'DarkOnBright',1,...
                  'MinArea',0.0002,...
                  'MaxArea',0.01); % 0.006

              
%% display detected keypoints on image
figure(1); clf;
imshow(I); hold on;
plot(f(2,:),f(1,:),'r.','LineWidth',3); % f_ = vl_ertr(f); vl_plotframe(f_);
plot(f(2,:),f(1,:),'go','LineWidth',1.5);
nf = size(f,2); % the number of detected keypoints
  

%% train the weight vector of the linear SVM classifier for nuclei detection
[w,b] = train_nuclei_classifier(0.1, rectified_bbox_dim, hogcell_dim);


%% find the tightest bounding boxes around the keypoints
% check all keypoints, and determine the optimal local bounding boxes
% around each keypoint
clear opt_vp opt_fval

for i = 1:nf
    
    x = f(1,i); % column
    y = f(2,i); % row
    s11 = f(3,i);
    s12 = f(4,i);
    s22 = f(5,i);
    if s11>2*s22
        s11 = 2*s22;
    elseif s22>2*s11
        s22 = 2*s11;
    end
    
    % plot(y,x,'r*');

    % optimization to find the local bounding box around each keypoint
    % that maximizes the score
    options = optimset('Algorithm','active-set',...
                       'TolFun',1e-8,...
                       'MaxFunEval',1000,...
                       'DiffMinChange',3);
                        % 'FinDiffRelStep',0.5,
                        % 'TolFun',1e-6,
                        % 'FinDiffRelStep',0.5
    
    % constraints:
    % -(x1-x0)<=-10 % -(y1-y0)<=-10 %  (x1-x0)<=50 %  (y1-y0)<=50
    % 0.5 <= (x1-x0)/(y1-y0) <=2
    % (x0+x1)/2 ~ x, (y0+y1)/2 ~ y
    eps = 10;
    Constraint.A = [1 0 -1 0; 0 1 0 -1; -1 0 1 0; 0 -1 0 1;...
                    1 -0.5 -1 0.5; -1 2 1 -2;
                    0.5 0 0.5 0; -0.5 0 -0.5 0;
                    0 0.5 0 0.5; 0 -0.5 0 -0.5];
    Constraint.b = [-11;-11;50;50; 0;0;
                    x+eps; -x+eps; y+eps; -y+eps];
    Constraint.lb = [max(1,x-40);max(1,y-40);x;y]; % [x-30; y-30; x; y];
    Constraint.ub = [x;y;min(512,x+40);min(512,y+40)]; % [x+30; y+30; 512; 512];
    Constraint.Aeq = []; %[0.5 0 0.5 0; 0 0.5 0 0.5];
    Constraint.beq = []; %[x;y];
    
    % repeat the optimization step with various initial bboxes
    min_fval = Inf;
    for del = [2 4 6 8]
%         bbox0 = [max(1,round(x-s11/del));
%                  max(1,round(y-s22/del));
%                  min(512,round(x+s11/del));
%                  min(512,round(y+s22/del))];
        bbox0 = [max(1,floor(x-s11/del));
                 max(1,floor(y-s22/del));
                 min(512,ceil(x+s11/del));
                 min(512,ceil(y+s22/del))];
        [vp,fval] = fmincon(@(bbox)compute_score(I,bbox,w,b,rectified_bbox_dim,hogcell_dim),...
            bbox0, Constraint.A, Constraint.b, Constraint.Aeq, Constraint.beq, Constraint.lb, Constraint.ub, [], options);
        
        fprintf('bbox:[%d %d %d %d], fval:%.4f\n',round(vp), fval);       
        out = ceil(vp);
        rectangle('Position',[out(2) out(1) out(4)-out(2) out(3)-out(1)])
        
        if fval < min_fval
            min_vp = vp;
            min_fval = fval;
        end
    end
    opt_pt(i,:) = [x y];
    opt_vp(i,:) = ceil(min_vp);
    opt_fval(i) = min_fval;
end

figure(2); clf;
imshow(I); hold on;
for i = 1:nf
    if opt_fval(i) >0.5
        continue;
    end
    plot(f(2,i),f(1,i),'r*');
    rectangle('Position',[opt_vp(i,2) opt_vp(i,1) ...
                         opt_vp(i,4)-opt_vp(i,2) opt_vp(i,3)-opt_vp(i,1)]);
end

idx = find(opt_fval > 0.5);
opt_pt(idx,:) = [];
opt_vp(idx,:) = [];
opt_fval(idx) = [];


%% final output
bbox_nuclei.center = opt_pt;
bbox_nuclei.bbox = opt_vp;
bbox_nuclei.fval = opt_fval;

 
%% segment by thresholding
% LL = bwlabel(ucm<=0.5); figure(7); imagesc(LL)
[BW, labels, num_cells] = segment_nuclei(bbox_nuclei, I); 

save(sprintf('../results/test%02d_nuclei.mat', test_id), ...
    'bbox_nuclei','BW','labels','num_cells','-v7.3');
% % display result
% figure(3); clf;
% subplot(211);
% imagesc(BW);
% title('Nuclei Segmentation Result');
% 
% BW_GT = D.test_gt.test_Nuclei{test_id};
% figure(3);
% subplot(212);
% imagesc(BW_GT);
% title('GT Segmentation');




function scores = compute_score(I,bbox,w,b,rectified_bbox_dim,hogcell_dim)

%% hog linear classifier test

bbox(1) = floor(bbox(1));
bbox(2) = floor(bbox(2));
bbox(3) = ceil(bbox(3));
bbox(4) = ceil(bbox(4));
% bbox = round(bbox);

patch = I(bbox(1):bbox(3),bbox(2):bbox(4));
if isempty(patch)
    scores = 100000;
end
patch = imresize(patch,[rectified_bbox_dim,rectified_bbox_dim]);

hog = vl_hog(single(patch), hogcell_dim);
xtest = hog(:);

scores = -double(w'*xtest + b);





%% extract SIFT keypoints
% figure(1); clf;
% imshow(max(gPb_orient,[],3))
% mgPb = max(gPb_orient,[],3);
%
% figure(2); clf; hold on; imshow(ucm)% imshow(I); hold on;
% peak_thresh = 2;
% edge_thresh = 10;
% f = vl_sift(single(mgPb*255),...
%             'PeakThresh',peak_thresh,'Edgethresh',edge_thresh);
% h = vl_plotframe(f);
% set(h,'color','b','linewidth',1);
%
%
% % filter keypoints
% for j = 1:size(f,2)
%     % linear classifier
% end
%
%
% % visualize image with defected SIFT keypoints
%
% I = D.test{end-10};
% figure(1); clf; imshow(I);
% [f,d] = vl_sift(single(I));
% sel = 1:length(f);
% h1 = vl_plotframe(f(:,sel)) ;
% h2 = vl_plotframe(f(:,sel)) ;
% set(h1,'color','k','linewidth',3) ;
% set(h2,'color','y','linewidth',2) ;
%
% figure(2); clf; imshow(I); hold on;
% peak_thresh = 3;
% edge_thresh = 5;
% f = vl_sift(single(I), 'PeakThresh', peak_thresh, ...
%                        'Edgethresh', edge_thresh);
%
% sel = find(f(3,:)>2 & f(3,:)<20);
% h = vl_plotframe(f(:,sel));
% set(h,'color','b','linewidth',2);