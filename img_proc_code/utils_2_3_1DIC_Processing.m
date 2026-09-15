% Image processing | Save in .h5 files for deep learning
% (1) input: oidic, 416x416, single
% (2) output: fluo, 416x416, single
clear; clc;

dataset_name = '10_2_4DIC_2d_arrange_LipidDroplet_live_training2_1DIC';
source_dir = ['./RAW/', dataset_name];
save_dir   = ['./Processed_', dataset_name];
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% % ---- (1) Train ----
% % num = num_train;
% data_dir = [source_dir, '/training'];
% save_path = [save_dir, '/train'];

% % ---- (2) Valid ----
% % num = num_valid;
% data_dir = [source_dir, '/validation'];
% save_path = [save_dir, '/valid'];

% ---- (3) Test ----
% num = num_test;
data_dir = [source_dir, '/testing'];
save_path = [save_dir, '/test'];


% ------------ Start ----------------
if ~exist(save_path, 'dir')  
    mkdir(save_path)
end


directory = dir(data_dir);
all_names = {directory.name};
folder_names = all_names(~ismember(all_names, {'.', '..'}));

for i=1:numel(folder_names)

    id = folder_names{i};
    id = int32(str2double(id));

    % Read
    dic_mat = single(imread([data_dir, '/', num2str(id), '/dic_1.tif']));
    fluo_mat = single(imread([data_dir, '/', num2str(id), '/fluo.tif']));

    % save to h5
    filename = [save_path, '/', num2str(id, '%04d'), '_dual.h5'];
    if exist(filename, 'file')
        delete(filename);
    end

    % h5write
    h5create(filename, '/dic1', [416, 416], 'DataType', 'single');
    h5create(filename, '/fluo',  [416, 416], 'DataType', 'single');

    h5write(filename, '/dic1', dic_mat);
    h5write(filename, '/fluo', fluo_mat);

    disp(i);

end
