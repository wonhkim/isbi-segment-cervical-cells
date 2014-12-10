% ? cell ??? ??? stability (energy)

% pairwise cells


function my_svm_struct_learn_ver2

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


cnt = 0;
for i = 1:num_trains
    
    num_cells = GT.CellNum(i);
    
    %% find a mapping between all cytoplasms and all nuclei
    % convert binary nuclei map into multi-label nuclei map
    labels_nuclei = bwlabel(GT.train_Nuclei{i},8);
    assert(length(unique(labels_nuclei))-1 == num_cells);
    % cytoplasm2nucleus(i): the nucleus label from labels_nuclei of i-th cell
    cytoplasm2nucleus = zeros(num_cells,1);
    
    for j = 1:num_cells
        % binary nuclei map inside the i-th cell
        bw_nuclei_in_cell = GT.train_Nuclei{i} & GT.train_Cytoplasm{i}{j};
        % query the list of corresponding labels from labels_nuclei
        idx = unique(labels_nuclei(bw_nuclei_in_cell));
        if length(idx) == 1
            cytoplasm2nucleus(j) = idx;
            continue;
        end
        k = 1;
        while length(idx)>1 && k<=length(idx)
            if sum(sum(labels_nuclei==idx(k))) ~= sum(sum(bw_nuclei_in_cell(labels_nuclei==idx(k))))
                idx(k) = [];
            else
                k = k + 1;
            end
        end
        if length(idx) == 1
            cytoplasm2nucleus(j) = idx;
            continue;
        end
    end
    
    cell_idx = find(cytoplasm2nucleus==0);
    nucleus_label = setdiff(1:num_cells,cytoplasm2nucleus);
    assert(length(cell_idx)==length(nucleus_label));
    if length(cell_idx) == 1
        cytoplasm2nucleus(cell_idx) = nucleus_label;
    elseif length(cell_idx) > 1
        dist_maps = cell(1,length(nucleus_label));
        for j = 1:length(nucleus_label)
            dist_maps{j} = bwdist(labels_nuclei==nucleus_label(j));
        end
        for k = 1:length(cell_idx)
            clear idx
            [idx(:,1),idx(:,2)] = find(GT.train_Cytoplasm{i}{cell_idx(k)}==1);
            center = round(mean(idx,1));
            min_d = Inf;
            for j = 1:length(nucleus_label)
                tmp_d = dist_maps{j}(center(1),center(2));
                if tmp_d < min_d
                    min_d = tmp_d;
                    cytoplasm2nucleus(cell_idx(k)) = nucleus_label(j);
                end
            end
        end
    end

    %%
    obj = load(sprintf('train%02d_ucm', i));
    
    for j = 1:num_cells
        cnt = cnt + 1;
        
        patterns{cnt} = double(Images{i});

        
        labels{cnt} = GT.train_Cytoplasm{i}{j};
        
        nuclei{cnt} = labels_nuclei==cytoplasm2nucleus(j);
        ucms{cnt} = obj.ucm;
        
        dist_maps{cnt} = bwdist(nuclei{cnt}, 'euclidean');
        bw = dist_maps{cnt} < 60 & dist_maps{cnt} > 0;
        
        gaussian_kernel = @(r,sig) exp(-r.^2/2/sig^2);
        prob_maps{cnt} = gaussian_kernel(dist_maps{cnt},25) .* bw;
        prob_maps{cnt} = prob_maps{cnt} / sum(sum(prob_maps{cnt}));
        
    end
end

% for id = 1:num_trains
%     % extract binary map for each nucleus
%     num_cells = GT.CellNum(id);
%     bw = cell(1,num_cells);
%     dist_map = cell(1,num_cells);
%     prob_map = cell(1,num_cells);
%     prob_maps{id} = double(zeros(dim));
%     for icell = 1:num_cells
%         bw{icell} = GT.train_Cytoplasm{id}{icell} & GT_nuclei{id}; % binary nucleus
%         L_new(bw{icell}) = icell; % segment label inside the cell nucleus
%         dist_map{icell} = bwdist(bw{icell}, 'euclidean');
%         bw_partial = dist_map{icell} < 60 & dist_map{icell} > 0;
% 
%         gaussian_kernel = @(r,sig) exp(-r.^2/2/sig^2);
%         prob_map{icell} = gaussian_kernel(dist_map{icell},30) .* bw_partial;
%         prob_map{icell} = prob_map{icell} / sum(sum(prob_map{icell}));
%         prob_maps{id} = prob_maps{id} + prob_map{icell};
%     end
% end

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
dbstop if error
save('result/ssvm-c_1-o_1-v_1-ver2.mat', 'model','-v7.3');

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

delta = sum(y(:) ~= ybar(:)); % the number of pixels having different labels
if param.verbose
    fprintf('In lossCB: loss is = %d\n', delta) ;
end
end

function psi = featureCB(param, x, y, prob_map, ucm, nuclei)


cytoplasm = (y & ~nuclei);
psi = zeros(3,1);

% maximize the probability to be involved in the cell
psi(1) = psi(1) + sum(sum(prob_map(cytoplasm)));

% minimize intensity variations inside the cell region
tmp = x(cytoplasm);
psi(2) = psi(2) - var(tmp(:));

% minimize the sum of ucm values
psi(3) = psi(3) - sum(sum(ucm(cytoplasm))) / sum(sum(cytoplasm));

% % intensity histogram
% xcenters = 50:30:230;
% h = hist(x(cytoplasm), xcenters) / sum(cytoplasm);
% psi = [psi; h'];

psi = sparse(psi);

if param.verbose
    fprintf('In featureCB: psi(x,y)\n');
end
end

function yhat = constraintCB(param, model, x, y, prob_map, ucm, nuclei)
% slack resaling: argmax_y delta(yi, y) (1 + <psi(x,y), w> - <psi(x,yi), w>)
% margin rescaling: argmax_y delta(yi, y) + <psi(x,y), w>
% if dot(y*x, model.w) > 1, yhat = y ; else yhat = - y ; end

% yhat: ?? score max? ??? GT ?? label y
% yhat = [];
y_list = generate_y_proposals(y, ucm, nuclei);
max_score = -Inf;
for i = 1:length(y_list) % size(y_list,2)
    % y_ = y_list(:,i);
    y_ = y_list{i};
    score = dot(featureCB(param,x,y_,prob_map,ucm,nuclei), model.w);
    if score > max_score
        max_score = score;
        yhat = y_;
    end
end

if param.verbose
    fprintf('In constraintCB: [%.4f, %.4f, %.4f]\n', model.w);
%     fprintf('yhat = violslack([%8.3f,%8.3f], [%8.3f,%8.3f], %3d) = %3d\n', ...
%         model.w, x, y, yhat) ;
end
end



function y_list = generate_y_proposals(y, ucm, nuclei)

% y = y(:);

dim = 512;
cytoplasm =  (y & ~nuclei); %(y & ~nuclei(:));

ucm_thres = 0.08;
superpixel_map = bwlabel(ucm<=ucm_thres, 8);

y_list = cell(0); % [];


%% find superpixels near the boundary

clear idx
[idx(:,1),idx(:,2)] = find(y==1); % ind2sub([dim,dim], find(y==1));

del = 10;
x0 = max(1,min(idx(:,1))-del); x1 = min(dim,max(idx(:,1))+del); w = x1-x0;
y0 = max(1,min(idx(:,2))-del); y1 = min(dim,max(idx(:,2))+del); h = y1-y0;
center = round([x0+x1, y0+y1]/2);
bbox = zeros(dim);
bbox(x0:x1,y0:y1) = 1;

% 1) consider additional superpixels
list_superpixels = unique( superpixel_map(bbox & ~cytoplasm) );
for k = 1:10:length(list_superpixels)
    y_ = y;
    for i = 1:k
        s = list_superpixels( randi([1,length(list_superpixels)],1) );
        clear idx
        y_(superpixel_map==s) = 1;
        y_(nuclei) = 1;
        y_ = fill_segments(y_,3);
        
%         if ~isequal(y,y_)
%             y_list = [y_list, y_];
%         end
    end
    if ~isequal(y,y_)
        y_list = [y_list, y_];
    end
end

% 2) remove superpixels
x0 = min(center(1)-50, x0 + 3*del); x1 = max(center(1)+50, x1 - 3*del);
y0 = min(center(2)-50, y0 + 3*del); y1 = max(center(2)+50, y1 - 3*del);
bbox = zeros(dim);
bbox(x0:x1,y0:y1) = 1;

list_superpixels = unique( superpixel_map(~bbox & cytoplasm) );
for k = 1:10:length(list_superpixels)
    y_ = y;
    for i = 1:k
        s = list_superpixels( randi([1,length(list_superpixels)],1) );
        clear idx
        y_(superpixel_map==s & ~nuclei) = 0;
        y_(nuclei) = 1;
        y_ = fill_segments(y_,3);

%         if ~isequal(y,y_)
%             y_list = [y_list,y_];
%         end
    end
    if ~isequal(y,y_)
        y_list = [y_list, y_];
    end
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

end