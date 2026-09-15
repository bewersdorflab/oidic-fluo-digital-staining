% Arrange dat: trainig, testing/validation
clear; clc;

% path
source_path = './RAW/10_1_4DIC_2d_nucleus_live_training/';
target_path = './RAW/10_2_4DIC_2d_arrange_nucleus_live_training/';


% training 22~570
sub_path = [target_path, '/training'];
for i = 22 : 570
    id = i;

    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_1.tif'], [sub_folder, '/dic_1.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_3.tif'], [sub_folder, '/dic_3.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_4.tif'], [sub_folder, '/dic_4.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_6.tif'], [sub_folder, '/dic_6.tif']);

    disp(i);
end


% testing 1~21, 571~596
sub_path = [target_path, '/testing'];
for i = 1 : 21
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_1.tif'], [sub_folder, '/dic_1.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_3.tif'], [sub_folder, '/dic_3.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_4.tif'], [sub_folder, '/dic_4.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_6.tif'], [sub_folder, '/dic_6.tif']);

    disp(i);
end


% validation 1~21, 571~596
sub_path = [target_path, '/validation'];
for i = 1 : 21
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_1.tif'], [sub_folder, '/dic_1.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_3.tif'], [sub_folder, '/dic_3.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_4.tif'], [sub_folder, '/dic_4.tif']);
    copyfile([source_path, '/dic/dic_', num2str(id), '_6.tif'], [sub_folder, '/dic_6.tif']);

    disp(i);
end
