clc; close all; clear all;

%%  constant
fault = 'Inner'; size = '1';
% fault = 'Normal'; size = '';
a = 1;
for iter = 3464: 3640 % 3464: 3640
    file_num = iter;
    file_name = strcat('D:\Downloads\SQdataset-main\', fault, '_', num2str(size), '\', 'REC', num2str(file_num), '_ch', num2str(2), '.txt');
    %     delete , '_', num2str(size) for normal
    if exist(file_name)
        data1 = readcell(strcat(file_name), Range=[17, 2]);

        data = cell2mat(data1);

        loc = 'D:\Downloads\SQdataset-main\mat_file\';
        data_name = strcat(loc, fault, '_', num2str(size), '_', num2str(a), '.mat');
        save(data_name, 'data');
        a = a+1;
    end
end


%%  varying
fault = 'OF'; size = '2';
% fault = 'N'; size = '';
a = 1;
len_data = [];
for iter = 3496
    file_num = iter;
    file_name = strcat('D:\Downloads\SQV-public\', fault, '_', num2str(size), '\', 'REC', num2str(file_num), '_ch', num2str(2), '.txt');
    %     delete , '_', num2str(size) for normal
    if exist(file_name)
        data1 = readcell(file_name, Range=[17, 2]);
        data2 = readcell(strcat('D:\Downloads\SQV-public\', fault, '_', num2str(size), '\', 'REC', num2str(file_num), '_ch', num2str(3), '.txt'), Range=[17, 2]);

        data1 = cell2mat(data1);
        data2 = -cell2mat(data2);

        [t1, speed, index] = get_speed(data2, 25.6e3, 5);
        figure,plot(t1, speed/60);
        set(gcf,'Position', [300,300,300, 220]);

        if ~isempty(index)
            data = data1(index(1): index(end));
            len_data = [len_data, index(end) - index(1)];

            loc = 'D:\Downloads\SQV-public\mat_file\';
            data_name = strcat(loc, fault, '_', num2str(size), '_', num2str(a), '.mat');
%             save(data_name, 'data');
            a = a+1;
        end


    end
end
tic 
toc

%%  get speed
[t1, speed, index] = get_speed(data2, 25.6e3, 0.2);
figure,plot(t1, speed/60);
set(gcf,'Position', [300,300,300, 220]);
set(gca,'Fontname', 'Times New Roman', 'Fontsize', 12);
xlabel('Time (s)'); ylabel('Frequency (Hz)');
axis tight;
ylim([0, 50])

fs = 25.6e3;
N = length(data2);
t = 1/fs: 1/fs: N/fs;
figure, plot(t(index(1): index(end)), data1(index(1): index(end)));
set(gcf,'Position', [300,300,300, 220]);
set(gca,'Fontname', 'Times New Roman', 'Fontsize', 12);
xlabel('Time (s)'); ylabel('Amplitude (m/s^2)');
axis tight;

figure,
plot(t1, speed/60);
set(gcf,'Position', [300,300,300, 220]);
