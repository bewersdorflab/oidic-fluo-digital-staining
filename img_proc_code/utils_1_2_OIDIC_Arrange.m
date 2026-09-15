% Arrange dat: trainig, testing/validation
clear; clc;

% path
source_path = './RAW/10_1_4DIC_2d_nucleus_live_training_OIDIC/';
target_path = './RAW/10_2_4DIC_2d_arrange_nucleus_live_training_OIDIC/';


% training
sub_path = [target_path, '/training'];
for i = 22 : 570
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end
    
    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/oidic/oidic_', num2str(id), '.tif'], [sub_folder, '/oidic.tif']);

    disp(i);
end


% testing
sub_path = [target_path, '/testing'];
for i = 1 : 21
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/oidic/oidic_', num2str(id), '.tif'], [sub_folder, '/oidic.tif']);

    disp(i);
end


% validation
sub_path = [target_path, '/validation'];
for i = 1 : 21
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/oidic/oidic_', num2str(id), '.tif'], [sub_folder, '/oidic.tif']);

    disp(i);
end

% testing
sub_path = [target_path, '/testing'];
for i = 571 : 596
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/oidic/oidic_', num2str(id), '.tif'], [sub_folder, '/oidic.tif']);

    disp(i);
end


% validation
sub_path = [target_path, '/validation'];
for i = 571 : 596
    id = i;
    
    sub_folder = [sub_path, '/', num2str(i)];
    if ~exist(sub_folder, 'dir')
        mkdir(sub_folder);
    end

    % Copy files
    copyfile([source_path, '/fluo/fluo_', num2str(id), '.tif'], [sub_folder, '/fluo.tif']);
    copyfile([source_path, '/oidic/oidic_', num2str(id), '.tif'], [sub_folder, '/oidic.tif']);

    disp(i);
end
