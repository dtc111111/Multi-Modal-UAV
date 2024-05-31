% Define the source directory and the destination directory
sourceDir = 'E:\anti_uav\train\';
destinationDir = 'E:\anti_uav\figure\png\time_plot\';

% Get a list of all directories in the source directory
sequenceDirs = dir(sourceDir);
sequenceDirs = sequenceDirs([sequenceDirs.isdir]); % Keep only directories
sequenceDirs = sequenceDirs(~ismember({sequenceDirs.name}, {'.', '..'})); % Remove '.' and '..'

% Loop through each directory
for i = 1:numel(sequenceDirs)
    % Extract sequence index from directory name
    seqIndex = sscanf(sequenceDirs(i).name, 'seq%d');
    
    % Check if the sequence index is valid
    if ~isempty(seqIndex)
        % Generate the source and destination file paths
        sourceFile = fullfile(sourceDir, sequenceDirs(i).name, 'figure/time_plot.png');
        destinationFile = fullfile(destinationDir, ['seq', num2str(seqIndex, '%02d'), '.png']);
        
        % Copy the file
        copyfile(sourceFile, destinationFile);
        
        % Display a message indicating the operation
        disp(['Copied ', sourceFile, ' to ', destinationFile]);
    end
end