function [BW, labels, num_cells] = segment_nuclei(bbox_nuclei, I)

dim = 512;
num_bboxes = length(bbox_nuclei.fval);


% BW = zeros(dim);
% for i = 1:num_bboxes
%     bbox = bbox_nuclei.bbox(i,:);
%     patch = I(bbox(1):bbox(3),bbox(2):bbox(4));
%     
%     mu = mean(mean(patch));
%     bw = patch < mu;
%     
%     if bbox_nuclei.fval(i) > 0.2
%         continue;
%     end
%     BW(bbox(1):bbox(3),bbox(2):bbox(4)) = BW(bbox(1):bbox(3),bbox(2):bbox(4)) | bw;
%     
% end

BW = zeros(dim);
for i = 1:num_bboxes
    bbox = bbox_nuclei.bbox(i,:);
    patch = I(bbox(1):bbox(3),bbox(2):bbox(4));
    
    h = size(patch,1);
    w = size(patch,2);
    bw = zeros(h,w);
    
    % quarter1
    mu = mean(mean(patch(1:ceil(h/2),1:ceil(w/2))));
    bw(1:ceil(h/2),1:ceil(w/2)) = patch(1:ceil(h/2),1:ceil(w/2)) < mu;
    
    % quarter2
    mu = mean(mean(patch(1:ceil(h/2),floor(w/2):end)));
    bw(1:ceil(h/2),floor(w/2):end) = patch(1:ceil(h/2),floor(w/2):end) < mu;    
    
    % quarter3
    mu = mean(mean(patch(floor(h/2):end,1:ceil(w/2))));
    bw(floor(h/2):end,1:ceil(w/2)) = patch(floor(h/2):end,1:ceil(w/2)) < mu;
    
    % quarter4
    mu = mean(mean(patch(floor(h/2):end,floor(w/2):end)));
    bw(floor(h/2):end,floor(w/2):end) = patch(floor(h/2):end,floor(w/2):end) < mu;

    if bbox_nuclei.fval(i) > 0.2
        continue;
    end
    BW(bbox(1):bbox(3),bbox(2):bbox(4)) = BW(bbox(1):bbox(3),bbox(2):bbox(4)) | bw;
    
end


labels = bwlabel(BW,4);
for i = 1:max(max(labels))
    if sum(sum(labels==i)) < 100
        labels(labels==i) = 0;
    end
end

labels = bwlabel(labels,4);
num_cells = max(max(labels));

iter = 10;
labels = fill_segments(labels,iter);
BW = labels > 0;

% %% post processing (cleaning segment labels)
% 
% % region marked as 1 but outside the nucleus
% idx = find(BW==1);
% [sub_idx(:,1),sub_idx(:,2)] = ind2sub([dim dim], idx);
% 
% for i = 1:length(idx)
%     % left, right, up, down
%     curr = sub_idx(i,:);
%     if ~isempty(find(curr==1 | curr==dim))
%         continue;
%     end
%     if BW(curr(1),curr(2)-1)==0 && ...% left
%        BW(curr(1),curr(2)+1)==0 && ...% right
%        BW(curr(1)-1,curr(2))==0 && ...% up
%        BW(curr(1)+1,curr(2))==0,      % down
%    
%         BW(idx(i)) = 0;
%     end
%    
% end

% % region inside the nucleus but marked as 0
% for i = 1:num_nuclei
%     idx = find(BW==1);
%     bbox = bbox_nuclei.bbox(i,:);
%     find(BW(bbox(1):bbox(3),bbox(2):bbox(4)
% end


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