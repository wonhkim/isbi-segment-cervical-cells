function my_svm_struct_learn

randn('state',0);
rand('state',0);

dim = 512;

% ------------------------------------------------------------------
% Train data
% ------------------------------------------------------------------

addpath('../BSR/grouping/lib');
addpath('../ISBI2014-overlapping_cervical_cells/Train45Test90')
addpath('../ISBI2014-overlapping_cervical_cells/Train45Test90/gpb_train');
addpath('../ISBI2014-overlapping_cervical_cells/code/train_segmentation'); % superpixel maps

object = load('isbi_train');
Images = object.ISBI_Train;
num_trains = length(Images);

GT = load('isbi_train_GT');
GT_nuclei = GT.train_Nuclei;


for i = 1:num_trains
    
    num_cells = GT.CellNum(i);
    
    patterns{i} = double(Images{i}(:)); % vectorize (columnize)
    labels{i} = double(zeros(dim));
    nuclei{i} = double(zeros(dim));
    
    % assign labels to each pixel - distinct labels for each individual cell
    for j = 1:num_cells
        labels{i}(GT.train_Cytoplasm{i}{j}) = j;
    end
    
    % assign labels to each pixel - intersection of different cells
    for j = 1:num_cells
        for k = j+1:num_cells
            labels{i}(GT.train_Cytoplasm{i}{j} & GT.train_Cytoplasm{i}{k}) = (j+k)/2;
        end
    end
    labels{i} = labels{i}(:); % vectorize (columnize)

    % ucm
    ucms{i} = load(sprintf('train%02d_ucm', i));
    ucms{i} = ucms{i}.ucm;
    
    % assign labels to each cell nucleus
    % nuclei{i} = double(GT_nuclei{i});
    for j = 1:num_cells
        nuclei{i}(GT.train_Nuclei{i} & GT.train_Cytoplasm{i}{j}) = j;
    end
    
end


for id = 1:num_trains
    % extract binary map for each nucleus
    num_cells = GT.CellNum(id);
    bw = cell(1,num_cells);
    dist_map = cell(1,num_cells);
    prob_map = cell(1,num_cells);
    prob_maps{id} = double(zeros(dim));
    for icell = 1:num_cells
        bw{icell} = GT.train_Cytoplasm{id}{icell} & GT_nuclei{id}; % binary nucleus
        L_new(bw{icell}) = icell; % segment label inside the cell nucleus
        dist_map{icell} = bwdist(bw{icell}, 'euclidean');
        bw_partial = dist_map{icell} < 60 & dist_map{icell} > 0;

        gaussian_kernel = @(r,sig) exp(-r.^2/2/sig^2);
        prob_map{icell} = gaussian_kernel(dist_map{icell},30) .* bw_partial;
        prob_map{icell} = prob_map{icell} / sum(sum(prob_map{icell}));
        prob_maps{id} = prob_maps{id} + prob_map{icell};
    end
end

% ------------------------------------------------------------------
%                                                    Run SVM struct
% ------------------------------------------------------------------

parm.patterns = patterns;
parm.labels = labels ;
parm.prob_maps = prob_maps;
parm.ucms = ucms;
parm.nuclei = nuclei;

parm.lossFn = @lossCB ;
parm.constraintFn  = @constraintCB ;
parm.featureFn = @featureCB ;

parm.dimension = 3;
parm.verbose = 1 ;

% fisrt input argument is learning parameters
% second input argument is structure learning parameters
model = svm_struct_learn(' -c 1.0 -o 1 -v 1 ', parm);
w = model.w ;

% ------------------------------------------------------------------
%                                                              Plots
% ------------------------------------------------------------------

% figure(1) ; clf ; hold on ;
% x = [patterns{:}] ;
% y = [labels{:}] ;
% plot(x(1, y>0), x(2,y>0), 'g.') ;
% plot(x(1, y<0), x(2,y<0), 'r.') ;
% set(line([0 w(1)], [0 w(2)]), 'color', 'y', 'linewidth', 4) ;
% xlim([-3 3]) ;
% ylim([-3 3]) ;
% set(line(10*[w(2) -w(2)], 10*[-w(1) w(1)]), ...
%     'color', 'y', 'linewidth', 2, 'linestyle', '-') ;
% axis equal ;
% set(gca, 'color', 'b') ;
% w
end

% ------------------------------------------------------------------
%                                               SVM struct callbacks
% ------------------------------------------------------------------

function delta = lossCB(param, y, ybar)
delta = sum(y ~= ybar); % the number of pixels having different labels
if param.verbose
    fprintf('In lossCB; loss is = %d\n', delta) ;
end
end

function psi = featureCB(param, x, y, prob_map, ucm, nuclei)
psi = zeros(3,1);

max_label = floor(max(y));

for i = 1:max_label
    
    idx_cell = unique([find(floor(y)==i); find(ceil(y)==i)]); % i-th cell region
    idx_nucleus = unique(find(nuclei==i));
    idx_cytoplasm = [];
    for j = 1:length(idx_cell)
        if isempty(find(idx_nucleus == idx_cell(j)))
            idx_cytoplasm = [idx_cytoplasm; idx_cell(j)];
        end
    end
    
    % psi(1)
    psi(1) = psi(1) + sum(prob_map(idx_cell));
    
    % psi(2)
    psi(2) = psi(2) - var(x(idx_cytoplasm));
    
    % psi(3)
    psi(3) = psi(3) - sum(ucm(idx_cell))/length(idx_cell);
    
    
end
psi = psi / max_label;
psi = sparse(psi);

if param.verbose
    fprintf('In featureCB; psi(x,y)\n');
end
end

function yhat = constraintCB(param, model, x, y, prob_map, ucm, nuclei)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
% if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end

% yhat: ?? score max? ??? GT ?? label y
yhat = [];
y_list = generate_yhat_rand(y, ucm, nuclei);
max_score = -Inf;
for i = 1:size(y_list,2)
    y_ = y_list(:,i);
    score = dot(featureCB(param,x,y_,prob_map,ucm,nuclei), model.w);
    if score > max_score
        max_score = score;
        yhat = y_;
    end
end

if param.verbose
    fprintf('In constraintCB; \n');
%     fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
%         model.w, x, y, yhat) ;
end
end




function y_list = generate_yhat_rand(y, ucm, nuclei)

dim = 512;
num_cells = max(max(nuclei));

k=0.05;
labels = bwlabel(ucm <= k); % superpixel map
% num_superpixels = max(max(labels));
L = reshape(y,dim,dim); % ground truth
% L_new = zeros(dim); % ground truth with slight perturbations

% make sure that nuclei labels equal to the gt cell labels in y
nuclei_ = nuclei;
for i = 1:num_cells
    idx = find(nuclei==i);
    if median(y(idx)) ~= i
        nuclei_(idx) = median(y(idx));
    end
end
nuclei = nuclei_;


% make changes to only one cell for each iter!
y_list = [];
list_dist_thres = 20:5:50; % distance transform around each nucleus

for icell = 1:num_cells
    for idist = length(list_dist_thres):-1:1
        
        dist_thres = list_dist_thres(idist);
        y_ = y;
        bw = (nuclei==icell); % binary map for the icell-th nucleus region
        dist_map = bwdist(bw, 'euclidean');
        
        % find all superpixels which is far from the nucleus more than the
        % "distance threshold", and currently(gt) included in the cell
            % bw_dist_map = (dist_map>dist_thres & L==icell);
        idx = find(dist_map>dist_thres & L==icell);
        list_superpixel_labels_in_the_region = unique(labels(idx));
        list_superpixel_labels_to_change = find(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1);
        for j = 1:length(list_superpixel_labels_to_change)
            y_(labels==j) = 0;
        end
        % for overlapping regions
        if icell>1
            % bw_dist_map = (dist_map>dist_thres & L==icell-0.5);
            idx = find(dist_map>dist_thres & L==icell-0.5);
            list_superpixel_labels_in_the_region = unique(labels(idx));
            y_(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1) = icell;
            list_superpixel_labels_to_change = find(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1);
            for j = 1:length(list_superpixel_labels_to_change)
                y_(labels==j) = icell;
            end
        end
        if icell<num_cells
            idx = find(dist_map>dist_thres & L==icell+0.5);
            list_superpixel_labels_in_the_region = unique(labels(idx));
            y_(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1) = icell;
            list_superpixel_labels_to_change = find(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1);
            for j = 1:length(list_superpixel_labels_to_change)
                y_(labels==j) = icell;
            end
        end
        if ~isequal(y,y_)
            y_list = [y_list, y_];
        end
    end
end
for idist = length(list_dist_thres):-1:1
    dist_thres = list_dist_thres(idist);
    y_ = y;
    for icell = 1:num_cells
        bw = (nuclei==icell); % binary map for the icell-th nucleus region
        dist_map = bwdist(bw, 'euclidean');
        
        % find all superpixels which is far from the nucleus more than the
        % "distance threshold", and currently(gt) included in the cell
        idx = find(dist_map>dist_thres & L==icell);
        list_superpixel_labels_in_the_region = unique(labels(idx));
        list_superpixel_labels_to_change = find(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1);
        for j = 1:length(list_superpixel_labels_to_change)
            y_(labels==j) = 0;
        end
        % for overlapping regions
        if icell>1
            % bw_dist_map = (dist_map>dist_thres & L==icell-0.5);
            idx = find(dist_map>dist_thres & L==icell-0.5);
            list_superpixel_labels_in_the_region = unique(labels(idx));
            list_superpixel_labels_to_change = find(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1);
            for j = 1:length(list_superpixel_labels_to_change)
                y_(labels==j) = icell;
            end
        end
        if icell<num_cells
            idx = find(dist_map>dist_thres & L==icell+0.5);
            list_superpixel_labels_in_the_region = unique(labels(idx));
            list_superpixel_labels_to_change = find(randi(2,1,length(list_superpixel_labels_in_the_region))-1 == 1);
            for j = 1:length(list_superpixel_labels_to_change)
                y_(labels==j) = icell;
            end
        end
    end
    if ~isequal(y,y_)
        y_list = [y_list, y_];
    end
end

% for i = 1:num_cells-1
%     
%     i_overlap = i + 0.5; % overlap region label
%     idx = find(y==i_overlap); % overlap region pixel linear indices
%     
%     % find all superpixels in the overlap region
%     list_superpixel_labels_in_overlap_region = unique(labels(idx));
%     list_superpixel_labels_to_change = ...
%         find(randi(2,1,length(list_superpixel_labels_in_overlap_region))-1 == 1);
%     y_ = y;
%     y_(list_superpixel_labels_to_change) = i;
%     
%     % check whether score is max
%     if score_ > score_max
%         score_ = score_max;
%         y_max = y_;
%     end
% end

% "initialize" superpixel labels
% - find the closest cell index for each superpixel
% - if sum of probabilities inside the superpixel is zero => background
% for s = 1:num_superpixels
%     prob_cells = zeros(1,num_cells);
%     for icell = 1:num_cells
%         prob_cells(icell) = sum(sum(prob_map(L==s)));
%     end
%     
%     % find the highest probability cell index and fill in the segment
%     % label to the superpixel
%     [max_prob,idx] = max(prob_cells);
%     if max_prob < 1e-4 || sum(sum(L==s)) > 5000
%         L_new(L==s) = 0; % background
%     else
%         L_new(L==s) = idx;
%     end
%     [~,idx] = find(superpixel2cell(s,:)==2); % nucleus
%     if length(idx)==1 % ~isempty(idx)
%         L_new(L==s) = idx;
%     end
%     
%     figure(7); imagesc(L_new);
% end

end