function segment_cervical_cells(test_id)

dim = 512;

%% load test data

addpath('~/Dropbox/Stanford/2014aut/cs279/project/ISBI2014-overlapping_cervical_cells/Train45Test90')
addpath('~/Dropbox/Stanford/2014aut/cs279/project/ISBI2014-overlapping_cervical_cells/Train45Test90/gpb_test');

object = load('isbi_test90');
I = object.ISBI_Test90{test_id};
% GT = load('isbi_test90_GT');
% GT_nuclei = GT.train_Nuclei;


%% load trained parameter
load('result/ssvm-c_1-o_1-v_1.mat');
w = model.w;

% % detect nuclei
% [bbox_nuclei] = detect_nuclei(test_id);
% 
% % determine the number of cells in the image
% [BW, labels, num_cells] = segment_nuclei(bbox_nuclei, I);

%%%%%%%%%%%%%%%%%%%%%%%%% 
%%%%% for debugging %%%%%
load(sprintf('test%02d_ucm.mat', test_id));
obj = load('isbi_test90_GT.mat');
nuclei_bw = obj.test_Nuclei{test_id};
cytoplasm_bw = obj.test_Cytoplasm{test_id};
num_cells = obj.CellNum(test_id);

nuclei_labels = bwlabel(nuclei_bw,4);
% nuclei_labels = zeros(dim);
% for i = 1:num_cells
%     nuclei_labels(cytoplasm_bw{i} & nuclei_bw) = i;
% end
%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%


%% compute prob_map
dist_map = bwdist(nuclei_bw, 'euclidean');
bw_partial = dist_map < 60 & dist_map > 0;
gaussian_kernel = @(r,sig) exp(-r.^2/2/sig^2);
prob_map = gaussian_kernel(dist_map,30) .* bw_partial;
prob_map = prob_map / sum(sum(prob_map));


%% initialize the segment labels
k=0.08;
[labels_init] = initialize_segment_labels(nuclei_labels, ucm, k, num_cells);
labels_init = fill_segments(labels_init, 5);
figure(7); imagesc(labels_init); title('labels init (k=0.08)');
 

%% segment by optimization (MAP)
parm.verbose = 0;  
ucm_thres = 0.08; 

list_labels = segmentation_proposals(@(x)w'*featureCB(parm,I(:),x,prob_map,ucm,nuclei_labels),...
    labels_init, nuclei_labels, ucm, ucm_thres);
 
% parm.verbose = 0;
opt_score = -Inf;
for iter = 1:size(list_labels,2)

    labels = list_labels(:,iter);
    psi = featureCB(parm, I(:), labels, prob_map, ucm, nuclei_labels);
    score = w'*psi;

    figure(3); imagesc(reshape(labels, dim, dim));
    title(score);
    
    if score > opt_score
        opt_score = score;
        opt_labels = labels;
    end
    figure(2); imagesc(reshape(opt_labels, dim, dim));
    title(opt_score);
end


disp(parm.verbose);



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
    if isempty(idx_cytoplasm)
        psi(2) = 1000;
    else
        psi(2) = psi(2) - var(single(x(idx_cytoplasm)));
    end
    
    % psi(3)
    psi(3) = psi(3) - sum(ucm(idx_cell))/length(idx_cell);
    
    
end
psi = psi / max_label;
psi = sparse(psi);

if param.verbose
    fprintf('In featureCB; psi(x,y)\n');
end