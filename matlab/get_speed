function [Time, Speed, index] = get_speed(speed_seq, fs, threshold)

N = length(speed_seq);
time = (1: N)/ fs;
tem = [];
index = [];
for iter = 1: N - 1
    if (speed_seq(iter) >= threshold) && (speed_seq(iter+1) <= threshold)
        tem = [tem, time(iter)];
        index = [index, iter];
    end
end

Speed = [];
Time = [];


for iiter = 2: length(tem)
    s_tem = 1/(tem(iiter) - tem(iiter - 1)) * 60;
    Speed = [Speed, s_tem];
    Time = [Time, tem(iiter)];
end
end
