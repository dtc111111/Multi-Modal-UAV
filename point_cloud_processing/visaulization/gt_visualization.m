clear;clc;
seq_folder = "train\seq1";
gt_folder = seq_folder + "\gt";
addpath(genpath(gt_folder))
addpath('npy-matlab') 
%b = readNPY('1706255121.954996.npy');

% Define the folder containing the .npy files

% Get a list of all .npy files in the folder
files = dir(fullfile(gt_folder, '*.npy'));


% Initialize variables for storing max and min values for each dimension
max_vals = [-inf, -inf, -inf];
min_vals = [inf, inf, inf];

% Loop through each file to find max and min values for each dimension
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
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

% Create figure with subplots
figure;

% Create 3D subplot
subplot(2, 2, 1);
hold on;
grid on;

% Create xy subplot
subplot(2, 2, 2);
hold on;
grid on;

% Create xz subplot
subplot(2, 2, 3);
hold on;
grid on;

% Create yz subplot
subplot(2, 2, 4);
hold on;
grid on;

% Loop through each file
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(gt_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Extract timestamp from filename
    [~, name, ~] = fileparts(filename);
    timestamp = str2double(name);
    
    % Plot the data in 3D subplot
    subplot(2, 2, 1);
    plot3(data(1), data(2), data(3), 'r.'); % Assuming 3D data
    
    % Set title with timestamp in 3D subplot
    title(sprintf('3D - Timestamp: %.3f', timestamp));
    
    % Plot the data in xy subplot
    subplot(2, 2, 2);
    plot(data(1), data(2), 'r.'); % Assuming 2D xy data
    
    % Set title with timestamp in xy subplot
    title(sprintf('XY - Timestamp: %.3f', timestamp));
    
    % Plot the data in xz subplot
    subplot(2, 2, 3);
    plot(data(1), data(3), 'r.'); % Assuming 2D xz data
    
    % Set title with timestamp in xz subplot
    title(sprintf('XZ - Timestamp: %.3f', timestamp));
    
    % Plot the data in yz subplot
    subplot(2, 2, 4);
    plot(data(2), data(3), 'r.'); % Assuming 2D yz data
    
    % Set title with timestamp in yz subplot
    title(sprintf('YZ - Timestamp: %.3f', timestamp));
    
    % Set axis limits for 3D subplot
    subplot(2, 2, 1);
    xlim(axis_limits(:, 1));
    ylim(axis_limits(:, 2));
    zlim(axis_limits(:, 3));
    
    % Set axis limits for xy subplot
    subplot(2, 2, 2);

    % Set axis limits for xz subplot
    subplot(2, 2, 3);

    
    % Set axis limits for yz subplot
    subplot(2, 2, 4);

    
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

saveas(gcf, 'gt_trajectory.fig');  % Save as .fig file

%%
% Create separate figures for time-x, time-y, and time-z plots
figure;
% Loop through each file
for i = 1:length(files)
    % Get the filename and full file path
    filename = files(i).name;
    filepath = fullfile(gt_folder, filename);
    
    % Load the data from the .npy file
    data = readNPY(filepath);
    
    % Extract timestamp from filename
    [~, name, ~] = fileparts(filename);
    timestamp = str2double(name);
    
    % Plot time-x, time-y, and time-z
    subplot(3, 1, 1);
    plot(timestamp, data(1), 'ro');
    hold on;
    xlabel('Time');
    ylabel('X');
    title('Time-X Plot');
    
    subplot(3, 1, 2);
    plot(timestamp, data(2), 'bo');
    hold on;
    xlabel('Time');
    ylabel('Y');
    title('Time-Y Plot');

    subplot(3, 1, 3);
    plot(timestamp, data(3), 'go');
    hold on;
    xlabel('Time');
    ylabel('Z');
    title('Time-Z Plot');
end

saveas(gcf, 'gt_time_plots.fig');  % Save as .fig file
% % Create figure
% figure;
% 
% 
% % Loop through each file
% for i = 1:length(files)
%     % Get the filename and full file path
%     filename = files(i).name;
%     filepath = fullfile(gt_folder, filename);
% 
%     % Load the data from the .npy file
%     data = readNPY(filepath);
% 
%     % Extract timestamp from filename
%     [~, name, ~] = fileparts(filename);
%     timestamp = str2double(name);
% 
%     % Plot the data
%     plot3(data(1), data(2), data(3), 'r.'); % Assuming 3D data
%     hold on;
%     grid on;
% 
%     % Set title with timestamp
%     title(sprintf('Timestamp: %.3f', timestamp));
% 
%     % Set axis limits based on max and min values
%     xlim(axis_limits(:, 1));
%     ylim(axis_limits(:, 2));
%     zlim(axis_limits(:, 3));
% 
% 
%     % Pause for the real time interval
%     if i < length(files)
%         % Calculate the time difference between timestamps
%         next_filename = files(i+1).name;
%         [~, next_name, ~] = fileparts(next_filename);
%         next_timestamp = str2double(next_name);
%         time_interval = next_timestamp - timestamp;
% 
%         % Pause for the time interval
%         pause(time_interval);
%     end
% end
% 
% hold off;
