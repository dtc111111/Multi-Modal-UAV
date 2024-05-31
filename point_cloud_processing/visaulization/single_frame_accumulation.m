clear;clc;
addpath('npy-matlab') 
dataset_path = "F:\anti_uav\train";
seq_path_list = dir(dataset_path);
save_npy_path = "F:\anti_uav\result\accumulation";
figure_path = "F:\anti_uav\figure\png\accumulation_plot";

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
        subplot(1, 2, 1);
        hold on;
        grid on;
        axis('equal')
        title('Avia')
        subplot(1, 2, 2);
        hold on;
        grid on;
        axis('equal')
        title('Mid 360')
        
        %% plot the avia
        disp("Processing Livox Avia")
        livox_avia_files = dir(fullfile(livox_avia_folder, '*.npy'));
        empty_count = 0;
        center_data_accumulation = [];
        for i = 1:length(livox_avia_files)
            filename = livox_avia_files(i).name;
            filepath = fullfile(livox_avia_folder, filename);
            data = readNPY(filepath);
            frame_num = str2double(filename(1:end-4));
            [gt_data, gt_frame_num] = find_gt(frame_num, gt_folder);
            mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
                data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
                data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            % mask = all(data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            data_filtered =  data(mask, :);
            if ~isempty(data_filtered)
                center_data = [data_filtered(:, 1)-gt_data(1), data_filtered(:, 2)-gt_data(2), data_filtered(:, 3)-gt_data(3)];
                center_data_accumulation = [center_data_accumulation;center_data];
            else
                empty_count = empty_count + 1;
            end
        end
        if ~isempty(center_data_accumulation)
            subplot(1, 2, 1);
            plot3(center_data_accumulation(:, 1), center_data_accumulation(:, 2) ,center_data_accumulation(:, 3), 'b.'); % XY view
        end
        writeNPY(center_data_accumulation, strcat(save_npy_path,'\avia\',seq_path_list(seq_ind).name,'.npy'))
        fprintf('%d / %d frames are empty\n',empty_count , length(livox_avia_files)) 


        %% plot the lidar 360
        disp("Processing Lidar 360")
        lidar_360_files = dir(fullfile(lidar_360_folder, '*.npy'));
        empty_count = 0;
        center_data_accumulation = [];
        for i = 1:length(lidar_360_files)
            filename = lidar_360_files(i).name;
            filepath = fullfile(lidar_360_folder, filename);
            data = readNPY(filepath);
            frame_num = str2double(filename(1:end-4));
            [gt_data, gt_frame_num] = find_gt(frame_num, gt_folder);
            mask = all(data(:,1) >= min_vals(1)-margin & data(:,1) <= max_vals(1)+margin & ...
                data(:,2) >= min_vals(2)-margin & data(:,2) <= max_vals(2)+margin & ...
                data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);
            % mask = all(data(:,3) >= min_vals(3)-margin & data(:,3) <= max_vals(3)+margin, 2);

            data_filtered =  data(mask, :);
           if ~isempty(data_filtered)
                center_data = [data_filtered(:, 1)-gt_data(1), data_filtered(:, 2)-gt_data(2), data_filtered(:, 3)-gt_data(3)];
                center_data_accumulation = [center_data_accumulation;center_data];
            else
                 empty_count = empty_count+1;
            end
        end
        if ~isempty(center_data_accumulation)
            subplot(1, 2, 2);
            plot3(center_data_accumulation(:, 1), center_data_accumulation(:, 2), center_data_accumulation(:, 3), 'g.'); % XY view
        end
        writeNPY(center_data_accumulation, strcat(save_npy_path,'\lidar_360\',seq_path_list(seq_ind).name,'.npy'))
        fprintf('%d / %d frames are empty\n',empty_count , length(lidar_360_files)) 
        % saveas(gcf, strcat(figure_path,'\trajectory.fig'));  % Save as .fig file
        saveas(gcf, strcat(figure_path,'\',seq_path_list(seq_ind).name,'.png'));  % Save as .fig file
        close(f)
    end
end



