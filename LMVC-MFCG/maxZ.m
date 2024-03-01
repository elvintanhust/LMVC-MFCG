clear;
clc;

load('cal101_7_Z_all.mat');
for i = 1:7
    for j=1:7
        Z = Z_all{i,j};
        [~,idx]=max(Z);
        idx_all{i,j} = idx;
    end
end