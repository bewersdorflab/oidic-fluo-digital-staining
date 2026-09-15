% 1DIC 2d
% Image Normalization/Processing
% Max-norm within [0, 1]
clear; clc;

% source path
source_path = './RAW/0_1_RAW/20250131_LipidDroplet_COS7_Live_Cell_Train2_1DIC';

% target_path
target_path = './RAW/10_1_4DIC_2d_LipidDroplet_live_training2_1DIC/';
target_fluo_path = [target_path, '/fluo'];
target_oidic_path = [target_path, '/dic'];
if ~exist(target_fluo_path, 'dir')
    mkdir(target_fluo_path);
end
if ~exist(target_oidic_path, 'dir')
    mkdir(target_oidic_path);
end

num_cases = 338; % total number of image pairs


% cases
for n = 1 : num_cases
    
    % Extract the max/min value
    max_dic = -inf; % Global max/min
    min_dic = inf;
    max_fluo = -inf;
    min_fluo = inf;
    
    img_dic = imread([source_path, '/img', num2str(n), '_dic1.tif']);
    img_fluo  = imread([source_path, '/img', num2str(n), '_fluo.tif']);
    
    min_dic = min(min_dic, min(img_dic(:))); % Global max/min
    max_dic = max(max_dic, max(img_dic(:)));
    min_fluo = min(min_fluo, min(img_fluo(:)));
    max_fluo = max(max_fluo, max(img_fluo(:)));
        
    % Norm 
    img_dic_norm = (img_dic - min_dic) / (max_dic - min_dic);
    img_fluo_norm = (img_fluo - min_fluo) / (max_fluo - min_fluo);
    % disp([min(img_oidic_norm(:)), max(img_oidic_norm(:)), min(img_fluo_norm(:)), max(img_fluo_norm(:))]);
    
    % Write
    func_tiffwrite(img_dic_norm, [target_path, '/dic/dic_', num2str(n), '_1.tif']);
    func_tiffwrite(img_fluo_norm, [target_path, '/fluo/fluo_', num2str(n), '.tif']);
    
    disp(['Case ', num2str(n, '%02d'), ' end']); 

end



