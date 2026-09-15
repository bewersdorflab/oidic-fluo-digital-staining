% Image processing | Save in .h5 files for deep learning
% (1) input: oidic, 416x416, single
% (2) output: fluo, 416x416, single
clear; clc;

dataset_name = '10_2_4DIC_2d_arrange_nucleus_live_training';
source_dir = ['./RAW/', dataset_name];
save_dir   = ['./Processed_', dataset_name];
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end

% % ---- (1) Train ----
num = num_train;
data_dir = [source_dir, '/training'];
save_path = [save_dir, '/train'];

% % ---- (2) Valid ----
% num = num_valid;
% data_dir = [source_dir, '/validation'];
% save_path = [save_dir, '/valid'];

% % ---- (3) Test ----
% num = num_valid;
% data_dir = [source_dir, '/testing'];
% save_path = [save_dir, '/test'];


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
    dic1_mat = single(imread([data_dir, '/', num2str(i), '/dic_1.tif']));
    dic3_mat = single(imread([data_dir, '/', num2str(i), '/dic_3.tif']));
    dic4_mat = single(imread([data_dir, '/', num2str(i), '/dic_4.tif']));
    dic6_mat = single(imread([data_dir, '/', num2str(i), '/dic_6.tif']));
    fluo_mat = single(imread([data_dir, '/', num2str(i), '/fluo.tif']));

    % save to h5
    filename = [save_path, '/', num2str(id, '%04d'), '_dual.h5'];
    if exist(filename, 'file')
        delete(filename);
    end

    % h5write
    h5create(filename, '/dic1', [416, 416], 'DataType', 'single');
    h5create(filename, '/dic3', [416, 416], 'DataType', 'single');
    h5create(filename, '/dic4', [416, 416], 'DataType', 'single');
    h5create(filename, '/dic6', [416, 416], 'DataType', 'single');
    h5create(filename, '/fluo',  [416, 416], 'DataType', 'single');

    h5write(filename, '/dic1', dic1_mat);
    h5write(filename, '/dic3', dic3_mat);
    h5write(filename, '/dic4', dic4_mat);
    h5write(filename, '/dic6', dic6_mat);
    h5write(filename, '/fluo', fluo_mat);

    disp(i);

end
