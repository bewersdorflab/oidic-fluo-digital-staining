% Arrange dat: trainig, testing/validation
clear; clc;

% path
source_path = './RAW/10_1_4DIC_2d_LipidDroplet_live_training2_1DIC/';
target_path = './RAW/10_2_4DIC_2d_arrange_LipidDroplet_live_training2_1DIC/';


% training
sub_path = [target_path, '/training'];
for i = 1 : 312
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end
    
    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_1.tif'], [sub_folder, '/dic_1.tif']);

    disp(i);
end


% testing
sub_path = [target_path, '/testing'];
for i = 313 : 338
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_1.tif'], [sub_folder, '/dic_1.tif']);

    disp(i);
end


% validation
sub_path = [target_path, '/validation'];
for i = 313 : 338
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_1.tif'], [sub_folder, '/dic_1.tif']);

    disp(i);
end


