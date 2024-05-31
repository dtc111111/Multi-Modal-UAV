function [gt_data, closest_gt_frame_num] = find_gt(livox_frame_num, gt_folder)
%UNTITLED3 此处显示有关此函数的摘要
%   此处显示详细说明
        frame_diff_min = inf;
        gt_files = dir(fullfile(gt_folder, '*.npy'));
        for i = 1:length(gt_files)
            filename = gt_files(i).name;
            gt_frame_num = str2double(filename(1:end-4));
            frame_diff = abs(livox_frame_num - gt_frame_num);
            if frame_diff < frame_diff_min
                closest_gt_filename = filename;
                closest_gt_frame_num = gt_frame_num;
                frame_diff_min = frame_diff;
            end
        end
        
        filepath = fullfile(gt_folder, closest_gt_filename);
        gt_data = readNPY(filepath);

end