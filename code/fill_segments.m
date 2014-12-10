function labels = fill_segments(labels, iter)

dim = 512;
num_segments = double(max(max(labels)));

for k = 1:iter % repeat many times
    
    for i = 1:num_segments
        
        clear lin_idx sub_idx
        lin_idx = find(labels == i);
        [sub_idx(:,1),sub_idx(:,2)] = ind2sub([dim dim], lin_idx);
        
        for j = 1:length(lin_idx)
            curr = lin_idx(j);
            up = get_pixel_up(curr,dim);
            down = get_pixel_down(curr,dim);
            left = get_pixel_left(curr,dim);
            right = get_pixel_right(curr,dim);
            
            if labels(up)==0 && (labels(get_pixel_up(up,dim))==i || labels(get_pixel_up(get_pixel_up(up,dim),dim))==i)
                labels(up) = i;
            end
            if labels(down)==0 && (labels(get_pixel_down(down,dim))==i || labels(get_pixel_down(get_pixel_down(down,dim),dim))==i)
                labels(down) = i;
            end
            if labels(left)==0 && (labels(get_pixel_left(left,dim))==i || labels(get_pixel_left(get_pixel_left(left,dim),dim))==i)
                labels(left) = i;
            end
            if labels(right)==0 && (labels(get_pixel_right(right,dim))==i || labels(get_pixel_right(get_pixel_right(right,dim),dim))==i)
                labels(right) = i;
            end
        end
    end
    
end

function out = get_pixel_up(idx, dim)
if mod(idx,dim) == 1
    out = idx; %0;
else
    out = idx - 1;
end

function out = get_pixel_down(idx, dim)
if mod(idx,dim) == 0
    out = idx; %0;
else
    out = idx + 1;
end

function out = get_pixel_left(idx, dim)
if idx>=1 && idx<=dim
    out = idx; %0;
else
    out = idx - dim;
end

function out = get_pixel_right(idx, dim)
if idx<=dim*dim && idx>=dim*(dim-1)+1
    out = idx; %0;
else
    out = idx + dim;
end