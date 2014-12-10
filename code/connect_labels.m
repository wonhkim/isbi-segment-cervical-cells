function y_new = connect_labels(y, superpixel_map)


% @description
%   The goal is to connect superpixels with the same label as one connected
%   region.
%
% @input
%   y : current segmentation map
%   superpixel_map : raw superpixel map
%
% @output
%   y_new : modified segmentation map where all superpixels with same
%   label consist of one connected region.
%
% @author
%   Wonhui Kim
%
% @contact
%   wonhui@stanford.edu


dim = 512;

max_label = max(max(y)); % # different regions except the background
% L = bwlabel(y,4); 
y_new = y;

% check for each segment label
for i = 1:max_label
    
    L = bwlabel(y==i, 4); % # of disjoint regions
    % connect all superpixels with label i
    idx = find(y==i);
    
    labels_disjoint = unique(L(idx));
    
    for li = 1:length(labels_disjoint)
        si = labels_disjoint(li);
        clear lin_idx_i sub_idx_i
        lin_idx_i = find(L==si);
        [sub_idx_i(:,1), sub_idx_i(:,2)] = ind2sub([dim dim], lin_idx_i);
        center_i = round(mean(sub_idx_i,1));
        
        for lj = li+1:length(labels_disjoint)
            sj = labels_disjoint(lj);
            clear lin_idx_j sub_idx_j
            lin_idx_j = find(L==sj);
            [sub_idx_j(:,1), sub_idx_j(:,2)] = ind2sub([dim dim], lin_idx_j);
            center_j = round(mean(sub_idx_j,1));
            
            % change labels of all superpixels near the path connecting the
            % two centers - center_i and center_j
            xmin = min(center_i(1), center_j(1));
            xmax = max(center_i(1), center_j(1));
            ymin = min(center_i(2), center_j(2));
            ymax = max(center_i(2), center_j(2));
            
            if xmin ~= xmax
                a = (ymax-ymin)/(xmax-xmin); % slope
                b = center_j(2) - a*center_j(1); % zero-crossing on y-axis
                % y = slope*x + b => center(2) = slope*center(1) + b
            
                for x = xmin:xmax
                    y = round(a*x + b);
                    s = superpixel_map(x,y);
                    if sum(sum(superpixel_map==s)) > 500
                        continue;
                    end
                    y_new(superpixel_map==s) = i;
                end
            else
                for y = ymin:ymax
                    s = superpixel_map(x,y);
                    if sum(sum(superpixel_map==s)) > 500
                        continue;
                    end
                    y_new(superpixel_map==s) = i;
                end
            end

            % pos = find(L==si | L==sj)
            % mean position ?? ?? ? ?? superpixel? ?? connect
        end
        
    end
end