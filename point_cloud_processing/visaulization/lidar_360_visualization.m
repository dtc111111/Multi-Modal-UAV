clear;clc;
seq_folder = "train\seq1";
lidar_360_folder = seq_folder + "\lidar_360";
addpath(genpath(lidar_360_folder))
addpath('npy-matlab') 
%b = readNPY('1706255121.954996.npy');

% Define the folder containing the .npy files

% Get a list of all .npy files in the folder
files = dir(fullfile(lidar_360_folder, '*.npy'));

%% set limits
gt_folder = seq_folder + "\gt";
gt_files = dir(fullfile(gt_folder, '*.npy'));

% Initialize variables for storing max and min values for each dimension
max_vals = [-inf, -inf, -inf];
min_vals = [inf, inf, inf];

% Loop through each file to find max and min values for each dimension
for i = 1:length(gt_files)
    % Get the filename and full file path
    filename = gt_files(i).name;
    filepath = fullfile(gt_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Update max and min values for each dimension
    max_vals = max(max_vals, data);
    min_vals = min(min_vals, data);
end

margin = 2;

% Calculate the center point of the trajectory
center_point = (max_vals + min_vals) / 2;

% Calculate the maximum range from the center point
max_range = max(max_vals - center_point) + margin;

% Set axis limits symmetrically around the center point
axis_limits = [center_point - max_range; center_point + max_range];


%% visualization
% Create figure with subplots
figure;

% Create 3D subplot
subplot(2, 2, 1);
hold on;
grid on;
xlim(axis_limits(:, 1));
ylim(axis_limits(:, 2));
zlim(axis_limits(:, 3));

% Create xy subplot
subplot(2, 2, 2);
hold on;
grid on;
xlim(axis_limits(:, 1));
ylim(axis_limits(:, 2));

% Create xz subplot
subplot(2, 2, 3);
hold on;
grid on;
xlim(axis_limits(:, 1));
ylim(axis_limits(:, 3));

% Create yz subplot
subplot(2, 2, 4);
hold on;
grid on;
xlim(axis_limits(:, 2));
ylim(axis_limits(:, 3));

% Loop through each file
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(lidar_360_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Extract timestamp from filename
    [~, name, ~] = fileparts(filename);
    timestamp = str2double(name);


    %% data filtering

    mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
        data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
        data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);

    data_filtered =  data(mask, :);

    if ~isempty(data)
    % Plot the 3D view
        subplot(2, 2, 1);
        plot3(data_filtered(:, 1), data_filtered(:, 2), data_filtered(:, 3), 'ro'); % Assuming 3D data
        title(sprintf('3D View - Timestamp: %.3f', timestamp));
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        grid on;
        
        % Plot the XY view
        subplot(2, 2, 2);
        plot(data_filtered(:, 1), data_filtered(:, 2), 'ro'); % XY view
        title('XY View');
        xlabel('X');
        ylabel('Y');
        grid on;
        
        % Plot the XZ view
        subplot(2, 2, 3);
        plot(data_filtered(:, 1), data_filtered(:, 3), 'ro'); % XZ view
        title('XZ View');
        xlabel('X');
        ylabel('Z');
        grid on;
        
        % Plot the YZ view
        subplot(2, 2, 4);
        plot(data_filtered(:, 2), data_filtered(:, 3), 'ro'); % YZ view
        title('YZ View');
        xlabel('Y');
        ylabel('Z');
        grid on;
        
    end
    % Pause for the real time interval
    if i < length(files)
        % Calculate the time difference between timestamps
        next_filename = files(i+1).name;
        [~, next_name, ~] = fileparts(next_filename);
        next_timestamp = str2double(next_name);
        time_interval = next_timestamp - timestamp;
        
        % Pause for the time interval
        pause(time_interval);
    end
end

saveas(gcf, 'lidar_360_trajectory.fig');  % Save as .fig file

%%
% Create separate figures for time-x, time-y, and time-z plots
figure;
subplot(3, 1, 1);
xlabel('Time');
ylabel('X');
title('Time-X Plot');

subplot(3, 1, 2);
xlabel('Time');
ylabel('Y');
title('Time-Y Plot');

subplot(3, 1, 3);
xlabel('Time');
ylabel('Z');
title('Time-Z Plot');

% Loop through each file
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(lidar_360_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
        data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
        data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);

    data_filtered =  data(mask, :);

    if ~isempty(data_filtered)
    % Extract timestamp from filename
    [~, name, ~] = fileparts(filename);
    timestamp = str2double(name);
    
    % Plot time-x, time-y, and time-z
    subplot(3, 1, 1);
    plot(timestamp, mean(data_filtered(:, 1)), 'ro');
    hold on;
    % ylim([min_vals(1)-1,max_vals(1)]);
    xlabel('Time');
    ylabel('X');
    title('Time-X Plot');
    
    subplot(3, 1, 2);
    plot(timestamp, mean(data_filtered(:, 2)), 'bo');
    hold on;
    % ylim([min_vals(2),max_vals(2)]);
    xlabel('Time');
    ylabel('Y');
    title('Time-Y Plot');

    subplot(3, 1, 3);
    plot(timestamp, mean(data_filtered(:, 3)), 'go');
    hold on;
    % ylim([min_vals(3),max_vals(3)]);
    xlabel('Time');
    ylabel('Z');
    title('Time-Z Plot');
    end
end
saveas(gcf, 'lidar_360_time_plots.fig');  % Save as .fig file
