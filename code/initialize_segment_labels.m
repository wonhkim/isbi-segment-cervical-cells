function [L_init] = initialize_segment_labels(L, ucm, ucm_thres, num_cells)

%
% @description
% initialize superpixel labels
%
% @input
% L: segment labels map for nuclei regions
% ucm: globalPb result
% ucm_thres: thereshold of ucm to get superpixels
% num_cells: the total number of cells in the image
%
% @contact
% Wonhui Kim (wonhui@stanford.edu)
%


% image dimension
dim = 512;


% get a superpixel map by thresholding the ucm
labels = bwlabel(ucm <= ucm_thres);
num_superpixels = max(max(labels));


% This is the matrix that will have the final segment labels.
% The k-th cell will have label k for its segments.
L_init = zeros(size(L));


% extract the binary maps, dist_maps, prob_maps for each nucleus
% bw{k} is the binary map for the k-th nucleus
% dist_map{k} is the distance map centered around the k-th nucleus
% prob_map{k} is the probability map, assumed to follow Gaussian
bw = cell(1,num_cells);
dist_maps = cell(1,num_cells);
prob_maps = cell(1,num_cells);
for icell = 1:num_cells
    bw{icell} = (L==icell);
    L_init(bw{icell}) = icell; % segment label inside the cell nucleus
    dist_maps{icell} = bwdist(bw{icell}, 'euclidean');
    bw_partial = dist_maps{icell} < 60 & dist_maps{icell} > 0;
    
    gaussian_kernel = @(r,sig) exp(-r.^2/2/sig^2);
    prob_maps{icell} = gaussian_kernel(dist_maps{icell},25) .* bw_partial;
    prob_maps{icell} = prob_maps{icell} / sum(sum(prob_maps{icell}));
end


% "initialize" superpixel labels
% - find the closest cell index for each superpixel
% - if sum of probabilities inside the superpixel is zero => background
for s = 1:num_superpixels
    
    % exclude background superpixels
    clear sub_idx
    [sub_idx(:,1), sub_idx(:,2)] = find(labels==s);
    
    % skip irrelevant superpixels
    if max(sub_idx(:,1)) - min(sub_idx(:,1)) > 100, continue; end
    if max(sub_idx(:,2)) - min(sub_idx(:,2)) > 100, continue; end
    if ~isempty(find(sub_idx==1 | sub_idx==dim,1)), continue; end
    if sum(sum(labels==s)) > 10000, continue; end % 5000
    if median(L(labels==s)) ~= 0, continue; end
    
    prob_cells = zeros(1,num_cells);
    for i = 1:num_cells
        prob_cells(i) = sum(sum(prob_maps{i}(labels==s)));
    end
    
    % find the highest probability cell index and fill in the segment
    % label to the superpixel
    [max_prob,idx] = max(prob_cells);
    if max_prob < 1e-4 || sum(sum(labels==s)) > 10000 %5000
        L_init(labels==s) = 0; % background
    else
        L_init(labels==s) = idx;
    end

    
end

% pixels in the k-th nucleus region must have label k
for i = 1:num_cells
    L_init(L==i) = i;
end




%     % find the closest cell index of s
%     min_dist = Inf;
%     for i = 1:num_cells
%         tmp_dist = mean(mean(dist_map{i}(labels==s)));
%         % tmp_dist = min(min(dist_map{i}(labels==s)));
%         % superpixel s is too far away from i-th nucleus
%         if tmp_dist > 200
%             closest_cell_idx = 0;
%             break;
%         end
%         if tmp_dist < min_dist % mean
%             min_dist = tmp_dist;
%             closest_cell_idx = i;
%         end
%     end
%     L_init(labels==s) = closest_cell_idx;