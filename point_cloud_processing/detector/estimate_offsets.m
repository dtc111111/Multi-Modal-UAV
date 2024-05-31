clear;clc;
addpath('npy-matlab') 
dataset_path = "E:\anti_uav\train";
seq_path_list = dir(dataset_path);
data_matrix_360 = [];
data_matrix_avia = [];
for seq_ind = 1:size(seq_path_list,1)
    if ~sum(strcmp(seq_path_list(seq_ind).name, {'.','..'}))
        seq_folder = strcat(dataset_path,'\',seq_path_list(seq_ind).name);
        disp("Processing " + seq_path_list(seq_ind).name)
        gt_folder = seq_folder + "\gt";
        addpath(genpath(gt_folder))
        lidar_360_folder = seq_folder + "\lidar_360";
        addpath(genpath(lidar_360_folder))
        livox_avia_folder = seq_folder + "\livox_avia";
        addpath(genpath(livox_avia_folder))
        radar_folder = seq_folder + "\radar_enhance_pcl";
        addpath(genpath(radar_folder))

        %% determine the ROI by gt with margin
        gt_files = dir(fullfile(gt_folder, '*.npy'));
        margin = 1; % 1m margin
        max_vals = [-inf, -inf, -inf];
        min_vals = [inf, inf, inf];
        for i = 1:length(gt_files)
            filename = gt_files(i).name;
            filepath = fullfile(gt_folder, filename);
            data = readNPY(filepath);
            max_vals = max(max_vals, data);
            min_vals = min(min_vals, data);
        end
        % for visualization
        center_point = (max_vals + min_vals) / 2;
        max_range = max(max_vals - center_point);
        axis_limits = [center_point - max_range-margin; center_point + max_range+margin];

    
      
        %% plot the ground truth
        disp("Processing Ground Truth")
        gt_files = dir(fullfile(gt_folder, '*.npy'));
        gt = [];
        for i = 1:length(gt_files)
            filename = gt_files(i).name;
            filepath = fullfile(gt_folder, filename);
            data = readNPY(filepath);
            [~, name, ~] = fileparts(filename);
            timestamp = str2double(name);
            gt = [gt;timestamp, data];
        end
        %% plot the avia
        disp("Processing Livox Avia")
        livox_avia_files = dir(fullfile(livox_avia_folder, '*.npy'));
        avia_lidar = [];
        for i = 1:length(livox_avia_files)
            filename = livox_avia_files(i).name;
            filepath = fullfile(livox_avia_folder, filename);
            data = readNPY(filepath);
            [~, name, ~] = fileparts(filename);
            timestamp = str2double(name);
            mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
                data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
                data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            % mask = all(data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            data_filtered =  data(mask, :);
            if ~isempty(data_filtered)
                avia_lidar = [avia_lidar; timestamp, mean(data_filtered, 1) ];
            end
        end
        
        %% plot the lidar 360
        disp("Processing Lidar 360")
        lidar_360_files = dir(fullfile(lidar_360_folder, '*.npy'));
        lidar_360 = [];
        for i = 1:length(lidar_360_files)
            filename = lidar_360_files(i).name;
            filepath = fullfile(lidar_360_folder, filename);
            data = readNPY(filepath);
            [~, name, ~] = fileparts(filename);
            timestamp = str2double(name);
            mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
                data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
                data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            % mask = all(data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            data_filtered =  data(mask, :);
            if ~isempty(data_filtered)
                lidar_360 = [lidar_360; timestamp, mean(data_filtered, 1) ];
            end
           
        end
    % Example data (replace with your actual data)


% Interpolate data2 and data3 to match time stamps of data1
if ~isempty(avia_lidar)
interp_avia = interp1(avia_lidar(:, 1), avia_lidar(:, 2:end), gt(:, 1), 'linear', 'extrap');
end
if ~isempty(lidar_360)
interp_360 = interp1(lidar_360(:, 1), lidar_360(:, 2:end), gt(:, 1), 'linear', 'extrap');
end


% Compute the residual data
if ~isempty(avia_lidar)
residual_avia = gt(:, 2:end) - interp_avia;
data_matrix_avia = [data_matrix_avia; gt, residual_avia];
end
if ~isempty(lidar_360)
residual_360 = gt(:, 2:end) - interp_360;
data_matrix_360 = [data_matrix_360; gt, residual_360];

end

        

    end
end
figure
hold on
plot(data_matrix_360(:,4), data_matrix_360(:,7),'rx')
plot(data_matrix_avia(:,4), data_matrix_avia(:,7),'bx')

residuals =  data_matrix_360(:,7);
residuals =  data_matrix_avia(:,7);

% Define the threshold for outlier detection
% Desired quantile values
quantiles = [0.25, 0.5, 0.75]; % Example quantile values

% Compute quantiles
quantile_values = quantile(residuals, quantiles);

% Create histogram bins based on quantiles
edges = [-Inf, quantile_values, Inf];

% Plot histogram
figure;
histogram(residuals, edges);
xlabel('Residuals');
ylabel('Frequency');
title('Histogram of Residuals with Quantiles');

