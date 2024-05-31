clear;clc;
addpath('npy-matlab') 
dataset_path = "E:\anti_uav\train";
seq_path_list = dir(dataset_path);

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
        %% figure initialization
        % Create figure with subplots
        f = figure;
        f.Visible = "on";
        
        % Create 3D subplot
        subplot(3, 1, 1);
        hold on;
        grid on;
        ylim(axis_limits(:, 1));
        xlabel('Time');
        ylabel('X');
        title('Time-X Plot');
        
        % Create xy subplot
        subplot(3, 1, 2);
        hold on;
        grid on;
        ylim(axis_limits(:, 2));
        xlabel('Time');
        ylabel('Y');
        title('Time-Y Plot');
        
        % Create tz subplot
        subplot(3, 1, 3);
        hold on;
        grid on;
        ylim(axis_limits(:, 3));
        xlabel('Time');
        ylabel('Z');
        title('Time-Z Plot');
        
      
        %% plot the ground truth
        disp("Processing Ground Truth")
        gt_files = dir(fullfile(gt_folder, '*.npy'));
        for i = 1:length(gt_files)
            filename = gt_files(i).name;
            filepath = fullfile(gt_folder, filename);
            data = readNPY(filepath);
            [~, name, ~] = fileparts(filename);
            timestamp = str2double(name);
            subplot(3, 1, 1);
            plot(timestamp, data(1), 'ro');
            subplot(3, 1, 2);
            plot(timestamp, data(2), 'ro');
            subplot(3, 1, 3);
            plot(timestamp, data(3), 'ro'); 
        end
        %% plot the avia
        disp("Processing Livox Avia")
        livox_avia_files = dir(fullfile(livox_avia_folder, '*.npy'));
        empty_count = 0;
        
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
                subplot(3, 1, 1);
                plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,1), 'b.');
                subplot(3, 1, 2);
                plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,2), 'b.');
                subplot(3, 1, 3);
                plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,3), 'b.'); 
            else
                empty_count = empty_count + 1;
            end
        end
        fprintf('%d / %d frames are empty\n',empty_count , length(livox_avia_files)) 
        
        %% plot the lidar 360
        disp("Processing Lidar 360")
        lidar_360_files = dir(fullfile(lidar_360_folder, '*.npy'));
        empty_count = 0;
        
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
                subplot(3, 1, 1);
                plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,1), 'g.');
                subplot(3, 1, 2);
                plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,2), 'g.');
                subplot(3, 1, 3);
                plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,3), 'g.'); 
            else
                 empty_count = empty_count+1;
            end
        end

        fprintf('%d / %d frames are empty\n',empty_count , length(lidar_360_files)) 
        
        
        %% plot the radar
        disp("Processing Radar")
        radar_files = dir(fullfile(radar_folder, '*.npy'));
        empty_count = 0;
        for i = 1:length(radar_files)
            filename = radar_files(i).name;
            filepath = fullfile(radar_folder, filename);
            data = readNPY(filepath);
            [~, name, ~] = fileparts(filename);
            timestamp = str2double(name);
            if ~isempty(data)
                mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
                    data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
                    data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
                % mask = all(data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
        
                data_filtered =  data(mask, :);
                if ~isempty(data_filtered)
                    subplot(3, 1, 1);
                    plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,1), 'y.');
                    subplot(3, 1, 2);
                    plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,2), 'y.');
                    subplot(3, 1, 3);
                    plot(timestamp*ones(size(data_filtered,1),1), data_filtered(:,3), 'y.'); 
                else
                    empty_count = empty_count+1;
                end
            else
                empty_count = empty_count+1;
            end
        end

        fprintf('%d / %d frames are empty\n',empty_count , length(radar_files)) 
        
        figure_path = strcat(seq_folder,'\figure');
        if exist(figure_path, 'dir') ~= 7
            mkdir(figure_path);
        end
        saveas(gcf, strcat(figure_path,'\time_plot.fig'));  % Save as .fig file
        saveas(gcf, strcat(figure_path,'\time_plot.png'));  % Save as .fig file
        close(f)

    end
end

