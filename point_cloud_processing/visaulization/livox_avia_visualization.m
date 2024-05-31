clear;clc;
seq_folder = "train\seq1";
livox_avia_folder = seq_folder + "\livox_avia";
addpath(genpath(livox_avia_folder))
addpath('npy-matlab') 
%b = readNPY('1706255121.954996.npy');

% Define the folder containing the .npy files

% Get a list of all .npy files in the folder
files = dir(fullfile(livox_avia_folder, '*.npy'));

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

% Calculate the center point of the trajectory
center_point = (max_vals + min_vals) / 2;

% Calculate the maximum range from the center point
max_range = max(max_vals - center_point);

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
    filepath = fullfile(livox_avia_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Extract timestamp from filename
    [~, name, ~] = fileparts(filename);
    timestamp = str2double(name);
    
    if ~isempty(data)
    % Plot the 3D view
        subplot(2, 2, 1);
        plot3(data(:, 1), data(:, 2), data(:, 3), 'ro'); % Assuming 3D data
        title(sprintf('3D View - Timestamp: %.3f', timestamp));
        xlabel('X');
        ylabel('Y');
        zlabel('Z');
        grid on;
        
        % Plot the XY view
        subplot(2, 2, 2);
        plot(data(:, 1), data(:, 2), 'ro'); % XY view
        title('XY View');
        xlabel('X');
        ylabel('Y');
        grid on;
        
        % Plot the XZ view
        subplot(2, 2, 3);
        plot(data(:, 1), data(:, 3), 'ro'); % XZ view
        title('XZ View');
        xlabel('X');
        ylabel('Z');
        grid on;
        
        % Plot the YZ view
        subplot(2, 2, 4);
        plot(data(:, 2), data(:, 3), 'ro'); % YZ view
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

saveas(gcf, 'livox_avia_trajectory.fig');  % Save as .fig file

%%
% Create separate figures for time-x, time-y, and time-z plots
figure;
% Loop through each file
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(livox_avia_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Create a binary mask to identify lines where coordinates are [0,0,0]
    mask = all(data == 0, 2);
    
    % Filter out points where coordinates are [0,0,0]
    filtered_data = data(~mask, :);

    if ~isempty(filtered_data)
    % Extract timestamp from filename
    [~, name, ~] = fileparts(filename);
    timestamp = str2double(name);
    
    % Plot time-x, time-y, and time-z
    subplot(3, 1, 1);
    plot(timestamp, mean(filtered_data(:, 1)), 'ro');
    hold on;
    ylim([min_vals(1)-1,max_vals(1)+1]);
    xlabel('Time');
    ylabel('X');
    title('Time-X Plot');
    
    subplot(3, 1, 2);
    plot(timestamp, mean(filtered_data(:, 2)), 'bo');
    hold on;
    ylim([min_vals(2),max_vals(2)+2]);
    xlabel('Time');
    ylabel('Y');
    title('Time-Y Plot');

    subplot(3, 1, 3);
    plot(timestamp, mean(filtered_data(:, 3)), 'go');
    hold on;
    ylim([min_vals(3)-1,max_vals(3)+1]);
    xlabel('Time');
    ylabel('Z');
    title('Time-Z Plot');
    end
end

saveas(gcf, 'livox_avia_time_plots.fig');  % Save as .fig file
