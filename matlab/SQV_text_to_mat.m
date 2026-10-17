clear;
clc;

root_dir = 'D:\Downloads\SQV-public';
fault = 'OF';
fault_size = '2';

% Normal
% fault = 'N';
% fault_size = '';

fs = 25.6e3;

if isempty(fault_size)
    condition = fault;
else
    condition = [fault, '_', fault_size];
end

input_dir = fullfile(root_dir, condition);
output_dir = fullfile(root_dir, 'mat_file', condition);

if ~isfolder(output_dir)
    mkdir(output_dir);
end

files = dir(fullfile(input_dir, 'REC*_ch2.txt'));
num_saved = 0;

for k = 1:numel(files)
    record_id = extractBetween(files(k).name, 'REC', '_ch2.txt');
    record_id = record_id{1};

    vibration_file = fullfile(input_dir, files(k).name);
    speed_file = fullfile(input_dir, ['REC', record_id, '_ch3.txt']);

    vibration = cell2mat(readcell(vibration_file, 'Range', [17, 2]));
    speed_signal = -cell2mat(readcell(speed_file, 'Range', [17, 2]));
    vibration = vibration(:);
    speed_signal = speed_signal(:);

    [~, ~, index] = get_speed(speed_signal, fs, 5);

    if ~isempty(index)
        first_sample = index(1);
        last_sample = index(end);
        data = vibration(first_sample:last_sample);

        output_file = fullfile(output_dir, ...
            sprintf('%s_%d.mat', condition, num_saved + 1));
        save(output_file, 'data', 'fs', 'record_id', ...
            'condition', 'first_sample', 'last_sample');
        num_saved = num_saved + 1;
    end
end

disp('done');
