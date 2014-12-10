function [y_list] = segmentation_proposals(compute_score, labels_init, nuclei, ucm, ucm_thres)

% @description
%   generate segmentation proposals around the initial segment labels
%
% @input
%   labels_init : initial segmentation map
%   nuclei : segmentation map of all nuclei
%   ucm : globalPb contour map result
%   ucm_thres : threshold of ucm to get the superpixel map
%
% @output
%   y_list : all the label combinations to search
%
% @author
%   Wonhui Kim
%
% @contact
%   wonhui@stanford.edu


dim = 512;
labels_filled = fill_segments(labels_init, 5);

% superpixel map
superpixel_map = bwlabel(ucm <= ucm_thres);
num_superpixels = max(max(superpixel_map));

% distance maps
num_cells = max(max(nuclei));
dist_maps = cell(num_cells,1);
for icell = 1:num_cells
    bw = (nuclei == icell);
    dist_maps{icell} = bwdist(bw, 'euclidean');
end


%% find difference between coarse superpixel map and the finer one
% k = [0.1 0.15];
k = 0.15;
[labels_k] = initialize_segment_labels(nuclei, ucm, k, num_cells);
labels_k = fill_segments(labels_k, 5);
superpixel_map_coarse = bwlabel(ucm <= k);
num_superpixels_coarse = max(max(superpixel_map_coarse));

idx = find(labels_k==0 & labels_filled~=0);
superpixel_labels = unique(superpixel_map(idx));


% decide whether or not each superpixel is better to be included as a cell,
% or better to be in the background
score_init = compute_score(labels_filled(:));
disp(score_init);

opt_score = -Inf;
opt_y = labels_init;
% for iter = 1:100
    % cell boundary around the background
for k = ceil(length(superpixel_labels)/2):ceil(length(superpixel_labels)/2)% ceil(length(superpixel_labels)/4):length(superpixel_labels) % randi([3,length(superpixel_labels)],1);
    for iter = 1:100
        y_ = labels_filled;
        for j = 1:k
            i = randi([2,length(superpixel_labels)],1);
            s = superpixel_labels(i);

            y_(superpixel_map==s) = 0;

            score = compute_score(y_(:));
            if score > opt_score
                opt_score = score;
                opt_y = y_;
                figure(1); imagesc(opt_y); title(opt_score);
            end
            
        end
    end
end

%% find all superpixels which are at the inter-cell boundaries

bdry_superpixels_cells = []; % boundary with other cells
adjacent_cells = [];
idx = unique(superpixel_map(opt_y>0));
num_superpixels = length(idx);
for i = 1:num_superpixels
    
    s = idx(i);
    if s==0, continue; end
    
    % exclude outside cell region superpixels
    clear lin_idx sub_idx
    lin_idx = find(superpixel_map==s);
    [sub_idx(:,1), sub_idx(:,2)] = ind2sub([dim dim], lin_idx);
    
    x0 = max(1,min(sub_idx(:,1))-1); x1 = min(dim,max(sub_idx(:,1))+1);
    y0 = max(1,min(sub_idx(:,2))-1); y1 = min(dim,max(sub_idx(:,2))+1);

    % for each superpixel, check whether it is adjacent to the background
    % ratio of background pixels (label==0) inside the bounding box around
    % the superpixel s
    area = (x1-x0+1)*(y1-y0+1); % bbox area
    center = sub2ind([dim,dim], round((x0+x1)/2), round((y0+y1)/2)); % bbox center (~ superpixel center)
    [center_sub(1),center_sub(2)] = ind2sub([dim dim],center);
    
    del = 2; x0=x0-del; x1=x1+del; y0=y0-del; y1=y1+del;
    curr_label = labels_filled(center);
    bbox_region = labels_filled(x0:x1,y0:y1);
    
    if labels_init(center) ~= 0 && ...
            sum(sum(superpixel_map==s)) < 2000 && ...
            sum(sum(superpixel_map==s))/sum(sum(labels_init==labels_init(center))) < 0.25

        ratio = sum(sum(bbox_region~=0 & bbox_region~=curr_label)) / area;
        if ratio > 0
            bdry_superpixels_cells = [bdry_superpixels_cells; s];
            
            tmp_idx = find(bbox_region~=curr_label & bbox_region~=0);
            tmp_label = unique(bbox_region(tmp_idx));
            max_count = 0;
            for kk = 1:length(tmp_label)
                tmp_count = sum(sum(bbox_region==tmp_label(kk)));
                if tmp_count > max_count
                    max_count = tmp_count;
                    max_label = tmp_label(kk);
                end
            end
            adjacent_cells = [adjacent_cells; max_label];
            continue;
            % labels_init(superpixel_map==s) = 1.5;
        end
    end
    
end

for k = ceil(length(bdry_superpixels_cells)/4):length(bdry_superpixels_cells)
    % cell boundary around the background
    % k = randi([3,length(bdry_superpixels_cells)],1);
    for iter = 1:100
        y_ = opt_y;
        for j = 1:k
            i = randi([2,length(bdry_superpixels_cells)],1);
            s = bdry_superpixels_cells(i);
 
            y_(superpixel_map==s) = adjacent_cells(i);
            y_ = connect_labels(y_, superpixel_map);
            y_ = fill_segments(y_,2);

            score = compute_score(y_(:));
            if score > opt_score
                opt_score = score;
                opt_y = y_;
                figure(1); imagesc(opt_y); title(opt_score);
            end
        end
    end
end


%% difference
for s = 1:num_superpixels_coarse
    
    % find superpixels outside cell region
    clear lin_idx sub_idx
    lin_idx = find(superpixel_map_coarse==s);
    [sub_idx(:,1), sub_idx(:,2)] = ind2sub([dim dim], lin_idx);
    
    x0 = max(1,min(sub_idx(:,1))-1); x1 = min(dim,max(sub_idx(:,1))+1);
    y0 = max(1,min(sub_idx(:,2))-1); y1 = min(dim,max(sub_idx(:,2))+1);
    
    if x1-x0 > 200, continue; end
    if y1-y0 > 200, continue; end
    if ~isempty(find(sub_idx==1 | sub_idx==dim, 1)), continue; end
    if sum(sum(superpixel_map_coarse==s)) > 5000, continue; end
    min_dist = Inf;
    for icell = 1:num_cells
        tmp_dist = min(min(dist_maps{icell}(superpixel_map_coarse==s)));
        if tmp_dist<min_dist, min_dist = tmp_dist; end
    end
    if min_dist > 200, continue; end
end

%%
% % find all superpixels which are adjacent to the background, or at the
% % boundary of different cells
% bdry_superpixels = []; % boundary with the background
% bdry_superpixels_cells = []; % boundary with other cells
% for s = 1:num_superpixels
%     
%     % exclude outside cell region superpixels
%     clear lin_idx sub_idx
%     lin_idx = find(superpixel_map==s);
%     [sub_idx(:,1), sub_idx(:,2)] = ind2sub([dim dim], lin_idx);
%     
%     x0 = max(1,min(sub_idx(:,1))-1); x1 = min(dim,max(sub_idx(:,1))+1);
%     y0 = max(1,min(sub_idx(:,2))-1); y1 = min(dim,max(sub_idx(:,2))+1);
%     
%     if x1-x0 > 100, continue; end
%     if y1-y0 > 100, continue; end
%     if ~isempty(find(sub_idx==1 | sub_idx==dim, 1)), continue; end
%     if sum(sum(superpixel_map==s)) > 5000, continue; end
%     min_dist = Inf;
%     for icell = 1:num_cells
%         tmp_dist = min(min(dist_maps{icell}(superpixel_map==s)));
%         if tmp_dist<min_dist, min_dist = tmp_dist; end
%     end
%     if min_dist > 200, continue; end
%     
%     % for each superpixel, check whether it is adjacent to the background
%     % ratio of background pixels (label==0) inside the bounding box around
%     % the superpixel s
%     area = (x1-x0+1)*(y1-y0+1); % bbox area
%     center = sub2ind([dim,dim], round((x0+x1)/2), round((y0+y1)/2)); % bbox center (~ superpixel center)
%     [center_sub(1),center_sub(2)] = ind2sub([dim dim],center);
%     ratio = sum(sum(labels_filled(x0:x1,y0:y1) == 0)) / area;
%     if ratio > 0.1
%         bdry_superpixels = [bdry_superpixels; s];
%     end
%     x0=x0-1; x1=x1+1; y0=y0-1; y1=y1+1;
%     if labels_init(center) ~= 0 && sum(sum(superpixel_map==s)) < 2000 && ...
%             sum(sum(superpixel_map==s))/sum(sum(labels_init==labels_init(center))) < 0.25
%         ratio = sum(sum(labels_filled(x0:x1,y0:y1) ~= 0 & ...
%             labels_filled(x0:x1,y0:y1) ~= labels_init(center))) / area;
%         if ratio > 0
%             bdry_superpixels_cells = [bdry_superpixels_cells; s];
%             continue;
%             % labels_init(superpixel_map==s) = 1.5;
%         end
%         ratio = sum(sum(labels_filled(center_sub(1)-10:center_sub(1)+10,center_sub(2)-10:center_sub(2)+10)~=0 & ...
%             labels_filled(center_sub(1)-10:center_sub(1)+10,center_sub(2)-10:center_sub(2)+10)~=labels_init(center))) / area;
%         if ratio > 0
%             bdry_superpixels_cells = [bdry_superpixels_cells; s];
%         end
%     end
%     
% end
% 
% 
% % initial score
% score_init = compute_score(labels_filled(:));
% 
% 
% % determine the boundary pixels between different cells
% num_bdry_superpixels_cells = length(bdry_superpixels_cells);
% opt_score = -Inf;
% opt_y = [];
% 
% % y = labels_filled(:);
% for iter = 1:100
%     
%     k = randi([5,ceil(num_bdry_superpixels_cells/2)],1);
%     
%     for iter2 = 1:1000
%         superpixel_indices = randi(num_bdry_superpixels_cells,1,k);
% 
%         y = labels_filled(:);
%         for j = 1:k
%             s = bdry_superpixels_cells(superpixel_indices(j));
%             y(superpixel_map == s) = randi(2*[1,num_cells],1)/2;
%         end
%         y = fill_segments(y, 2);
%         score = compute_score(y);
%         % fprintf('init score is %.4f, score is %.4f\n', score_init, score);
%         if score > opt_score
%             opt_score = score;
%             opt_y = y;
%         end
%         % figure(5); imagesc(reshape(y,dim,dim)); title(score);
%         figure(3); imagesc(reshape(opt_y,dim,dim)); title(opt_score);
%     end
%     fprintf('iter %05d done!\n', iter);
% end
% 
% 
% % determine whether to maintain superpixels or not
% num_bdry_superpixels = length(bdry_superpixels);
% % for s = 1:num_bdry_superpixels
% %     y = labels_init(:);
% %     y(superpixel_map == bdry_superpixels(s)) = 0; % change pixel to background
% %     score = compute_score(y);
% %     fprintf('init score is %.4f, score is %.4f\n', score_init, score);
% %     figure(5); imagesc(reshape(y,dim,dim));
% % end
% opt_score = -Inf;
% opt_y = [];
% for k = 3:3:ceil(num_bdry_superpixels/3)
%     % choose k among all superpixels
%     combinations = nchoosek(bdry_superpixels, k);
%     disp(length(combinations));
%     for i = 3:3:length(combinations)
%         superpixel_indices = combinations(i,:);
%         y = labels_init(:);
%         y_ = y;
%         for j = 1:length(superpixel_indices)
%             s = superpixel_indices(j);
%             y(superpixel_map == s) = 0;
%             y_(superpixel_map == s) = 0;
%             if ~isempty(find(bdry_superpixels_cells==s, 1))
%                 y_(superpixel_map == s) = 1.5;
%             end
%         end
%         score = compute_score(y);
%         score_ = compute_score(y_);
%         fprintf('init score is %.4f, score is %.4f\n', score_init, score);
%         % figure(5); imagesc(reshape(y,dim,dim)); title(score);
%         if ~isequal(y,y_) && score_ > opt_score
%             opt_score = score_;
%             opt_y = y_;
%         end
%         if score > opt_score
%             opt_score = score;
%             opt_y = y;
%         end
%         figure(3); imagesc(reshape(opt_y,dim,dim)); title(opt_score);
%     end
% end





%% previous version
% % make changes to only one cell for each iter!
% num_cells = max(max(nuclei));
% y = labels_init(:);
% 
% % list_dist_thres = dist_lb:5:dist_ub; % 20:5:50; % distance transform around each nucleus
% list_superpixel_labels_in_the_region = [];
% dist_maps = cell(0);
% for icell = 1:num_cells
% 
%     % find all superpixels which are within the range of distance lb~ub
%     % from the nucleus
%     bw = (nuclei == icell); % binary map for the icell-th nucleus region
%     dist_map = bwdist(bw, 'euclidean');
%     
%     idx = find(dist_map > dist_lb & dist_map <= dist_ub);
%     list_superpixel_labels_in_the_region = ...
%         [list_superpixel_labels_in_the_region; unique(superpixel_map(idx))];
% 
%     dist_maps{icell} = dist_map;
% end
% 
% num_target_superpixels = length(list_superpixel_labels_in_the_region);
% idx = [];
% for i = 1:num_target_superpixels
%     s = list_superpixel_labels_in_the_region(i);
%     if s == 0
%         idx = [idx; i];
%     end
%     if sum(sum(superpixel_map == s)) > 5000
%         idx = [idx; i];
%     end
% end
% list_superpixel_labels_in_the_region(idx) = [];
% 
% num_target_superpixels = length(list_superpixel_labels_in_the_region);
% y_list = [];
% for k = 4:3:num_target_superpixels
%     
%     % choose k among all superpixels
%     combinations = nchoosek(1:num_target_superpixels, k);
%     disp(length(combinations));
%     
%     for i = 4:3:length(combinations)
%         y_ = y;
%         superpixel_indices = combinations(i,:);
%         for j = 1:length(superpixel_indices)
%             s = list_superpixel_labels_in_the_region(superpixel_indices(j));
%             
%             % find the closest cell index of s
%             min_dist = Inf;
%             min_dist2cells = zeros(num_cells,1);
%             for icell = 1:num_cells
%                 min_dist2cells(icell) = min(dist_maps{icell}(superpixel_map==s)); % mean
%             end
%              
%             superpixel_label = median(y(superpixel_map == s));
%             eps = 5;
%             if superpixel_label == 0
%                 y_(superpixel_map == s) = closest_cell_idx; % the closest cell index
%                 continue;
%             end
%             if superpixel_label>1
%                 if abs(min_dist2cells(superpixel_label)-min_dist2cells(superpixel_label-1)) < eps
%                     y_(superpixel_map == s) = superpixel_label - 0.5;
%                     continue;
%                 end
%             end
%             if superpixel_label < num_cells
%                 if abs(min_dist2cells(superpixel_label)-min_dist2cells(superpixel_label+1)) < eps
%                     y_(superpixel_map == s) = superpixel_label + 0.5;
%                     continue;
%                 end
%             end
%             y_(superpixel_map == s) = 0;
% 
%         end
%         if ~isequal(y,y_)
%             y_list = [y_list, y_];
%         end
%     end
%     y_list = sparse(y_list);
%     save('test01_label_list.mat', 'y_list','-v7.3');
%     y_list = [];
% end
% % for icell = 1:num_cells
% %     % check whether each of selected superpixels already has correct label
% %     for i = 1:length(list_superpixel_labels_in_the_region)
% %         y_ = y;
% %         
% %         s = list_superpixel_labels_in_the_region(i);
% %         if s == 0
% %             continue;
% %         end
% %         if sum(sum(superpixel_map == s)) > 5000
% %             continue;
% %         end
% %         if median(y(superpixel_map == s)) == 0
% %             y_(superpixel_map == s) = icell;
% %         elseif median(y(superpixel_map == s)) == icell
% %             y_(superpixel_map == s) = 0;
% %         elseif median(y(superpixel_map == s)) == icell+1
% %             y_(superpixel_map == s) = icell+0.5;
% %         elseif median(y(superpixel_map == s)) == icell-1
% %             y_(superpixel_map == s) = icell-0.5;
% %         else
% %             continue;
% %         end
% %         if ~isequal(y,y_)
% %             y_list = [y_list, y_];
% %         end
% %     end
% %     
% %     y_ = y;
% %     for i = 1:length(list_superpixel_labels_in_the_region)
% %         s = list_superpixel_labels_in_the_region(i);
% %         if s == 0
% %             continue;
% %         end
% %         if sum(sum(superpixel_map == s)) > 5000
% %             continue;
% %         end
% %         if median(y(superpixel_map == s)) == 0
% %             y_(superpixel_map == s) = icell;
% %         elseif median(y(superpixel_map == s)) == icell
% %             y_(superpixel_map == s) = 0;
% %         elseif median(y(superpixel_map == s)) == icell+1
% %             y_(superpixel_map == s) = icell+0.5;
% %         elseif median(y(superpixel_map == s)) == icell-1
% %             y_(superpixel_map == s) = icell-0.5;
% %         else
% %             continue;
% %         end
% %         if ~isequal(y,y_)
% %             y_list = [y_list, y_];
% %         end
% %     end
% %     
% %     
% % end