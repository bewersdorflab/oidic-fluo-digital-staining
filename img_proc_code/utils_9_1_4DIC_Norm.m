% 4DIC 2d
% Image Normalization/Processing
% Max-norm within [0, 1]
clear; clc;

% source path
source_path = './RAW/0_1_RAW/20250116_Hoechst33342_COS7_Live_Cell_Train';

% target_path
target_path = './RAW/10_1_4DIC_2d_nucleus_live_training/';
target_fluo_path = [target_path, '/fluo'];
target_dic_path = [target_path, '/dic'];
if ~exist(target_fluo_path, 'dir')
    mkdir(target_fluo_path);
end
if ~exist(target_dic_path, 'dir')
    mkdir(target_dic_path);
end

num_cases = 338;    % total number of image pairs


% cases
for n = 1 : num_cases
    
    % Extract the max/min value
    max_dic = -inf; % Global max/min
    min_dic = inf;
    max_fluo = -inf;
    min_fluo = inf;
    
    img_dic1 = imread([source_path, '/img', num2str(n), '_dic1.tif']);
    img_dic3 = imread([source_path, '/img', num2str(n), '_dic3.tif']);
    img_dic4 = imread([source_path, '/img', num2str(n), '_dic4.tif']);
    img_dic6 = imread([source_path, '/img', num2str(n), '_dic6.tif']);
    img_fluo = imread([source_path, '/img', num2str(n), '_fluo.tif']);
        
    max_dic_frame = max([max(img_dic1(:)), max(img_dic3(:)), max(img_dic4(:)), max(img_dic6(:))]);
    min_dic_frame = min([min(img_dic1(:)), min(img_dic3(:)), min(img_dic4(:)), min(img_dic6(:))]);
    max_dic = max(max_dic, max_dic_frame);
    min_dic = min(min_dic, min_dic_frame);

    min_fluo = min(min_fluo, min(img_fluo(:)));
    max_fluo = max(max_fluo, max(img_fluo(:)));

    img_dic1_norm = (img_dic1 - min_dic) / (max_dic - min_dic);
    img_dic3_norm = (img_dic3 - min_dic) / (max_dic - min_dic);
    img_dic4_norm = (img_dic4 - min_dic) / (max_dic - min_dic);
    img_dic6_norm = (img_dic6 - min_dic) / (max_dic - min_dic);
    img_fluo_norm = (img_fluo - min_fluo) / (max_fluo - min_fluo);
    
    func_tiffwrite(img_dic1_norm, [target_path, '/dic/dic_', num2str(n), '_1.tif']);
    func_tiffwrite(img_dic3_norm, [target_path, '/dic/dic_', num2str(n), '_3.tif']);
    func_tiffwrite(img_dic4_norm, [target_path, '/dic/dic_', num2str(n), '_4.tif']);
    func_tiffwrite(img_dic6_norm, [target_path, '/dic/dic_', num2str(n), '_6.tif']);
    func_tiffwrite(img_fluo_norm, [target_path, '/fluo/fluo_', num2str(n), '.tif']);
        
    disp(['Case ', num2str(n, '%02d'), ' end']); 

end



