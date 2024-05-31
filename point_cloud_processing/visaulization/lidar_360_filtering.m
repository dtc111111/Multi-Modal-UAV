clear;clc;
% seq_folder = "train\seq1";
seq_folder = "D:\val\val\seq0001";
lidar_360_folder = seq_folder + "\lidar_360";
output_folder = seq_folder+'\lidar_360_filtered';   % Folder to save filtered points
% Create the output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end
addpath(genpath(lidar_360_folder))
addpath('npy-matlab') 

% Define the folder containing the .npy files

% Get a list of all .npy files in the folder
files = dir(fullfile(lidar_360_folder, '*.npy'));


% Sort the files based on their timestamps
[~, sorted_indices] = sort([files.datenum]);
files = files(sorted_indices);

%% occupancy grid map
% Initialize variables for storing max and min values for each dimension
max_vals = [-inf, -inf, -inf];
min_vals = [inf, inf, inf];

% Loop through each file to find max and min values for each dimension
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(lidar_360_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Update max and min values for each dimension
    max_vals = max(max_vals, max(data));
    min_vals = min(min_vals, min(data));
end
% min_vals = [-5,-5 ,0];
% max_vals = [5, 5, 10];
% Define the grid size
grid_size = [1, 1, 50]; % Adjust this based on your requirements

% Determine the number of grid cells in each axis
num_cells_x = ceil((max_vals(1) - min_vals(1)) / grid_size(1));
num_cells_y = ceil((max_vals(2) - min_vals(2)) / grid_size(2));
num_cells_z = ceil((max_vals(3) - min_vals(3)) / grid_size(3));

% Initialize the occupancy grid map
occupancy_grid = zeros(num_cells_x, num_cells_y, num_cells_z);


% Loop through each file
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(lidar_360_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Iterate through the data points and count them in the appropriate grid cell
    for j = 1:size(data, 1)
        % Determine the grid cell index for the current point
        cell_index_x = ceil((data(j, 1) - min_vals(1)) / grid_size(1));
        cell_index_y = ceil((data(j, 2) - min_vals(2)) / grid_size(2));
        cell_index_z = ceil((data(j, 3) - min_vals(3)) / grid_size(3));
        
        % Ensure the point is within the grid bounds
        cell_index_x = max(min(cell_index_x, num_cells_x), 1);
        cell_index_y = max(min(cell_index_y, num_cells_y), 1);
        cell_index_z = max(min(cell_index_z, num_cells_z), 1);
        
        % Increment the count for the corresponding grid cell
        occupancy_grid(cell_index_x, cell_index_y, cell_index_z) = ...
        occupancy_grid(cell_index_x, cell_index_y, cell_index_z) + 1;
    end
    occupancy_grid = occupancy_grid * 0.5;
end

% Find occupied voxels
threshold =  1;
[occupied_x, occupied_y, occupied_z] = ind2sub(size(occupancy_grid), find(occupancy_grid >threshold));
[dynamic_x, dynamic_y, dynamic_z] = ind2sub(size(occupancy_grid), find(occupancy_grid <= threshold & occupancy_grid>0));

% Visualize the occupied voxels
figure(1)
subplot(1,2,1)
scatter3(occupied_x, occupied_y, occupied_z, 20, 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Occupied Voxels');
subplot(1,2,2)
scatter3(dynamic_x, dynamic_y, dynamic_z, 20, 'filled');
xlabel('X');
ylabel('Y');
zlabel('Z');
title('Dynamic Voxels');