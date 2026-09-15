% OIDIC 2d
% Image Normalization/Processing
% Max-norm within [0, 1]
clear; clc;

% source path
source_path = './RAW/0_1_RAW/20250116_Hoechst33342_COS7_Live_Cell_Train_OIDIC';

% target_path
target_path = './RAW/10_1_4DIC_2d_nucleus_live_training_OIDIC/';
target_fluo_path = [target_path, '/fluo'];
target_oidic_path = [target_path, '/oidic'];
if ~exist(target_fluo_path, 'dir')
    mkdir(target_fluo_path);
end
if ~exist(target_oidic_path, 'dir')
    mkdir(target_oidic_path);
end

num_cases = 596; % total number of image pairs


% cases
for n = 1 : num_cases
    
    % Extract the max/min value
    max_oidic = -inf; % Global max/min
    min_oidic = inf;
    max_fluo = -inf;
    min_fluo = inf;
    
    img_oidic = imread([source_path, '/img', num2str(n), '_oidic.tif']);
    img_fluo  = imread([source_path, '/img', num2str(n), '_fluo.tif']);
    
    min_oidic = min(min_oidic, min(img_oidic(:))); % Global max/min
    max_oidic = max(max_oidic, max(img_oidic(:)));
    min_fluo = min(min_fluo, min(img_fluo(:)));
    max_fluo = max(max_fluo, max(img_fluo(:)));
        
    % Norm 
    img_oidic_norm = (img_oidic - min_oidic) / (max_oidic - min_oidic);
    img_fluo_norm = (img_fluo - min_fluo) / (max_fluo - min_fluo);
    % disp([min(img_oidic_norm(:)), max(img_oidic_norm(:)), min(img_fluo_norm(:)), max(img_fluo_norm(:))]);
    
    % Write
    func_tiffwrite(img_oidic_norm, [target_path, '/oidic/oidic_', num2str(n), '.tif']);
    func_tiffwrite(img_fluo_norm, [target_path, '/fluo/fluo_', num2str(n), '.tif']);
    
    disp(['Case ', num2str(n, '%02d'), ' end']); 

end



