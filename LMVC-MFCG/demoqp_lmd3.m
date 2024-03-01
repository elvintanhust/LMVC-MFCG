clear;
clc;
warning off;
addpath(genpath('./'));

%% dataset
ds = {'Caltech101-7_Per1'}

dsPath = './0-dataset/';
resPath = './res-lmd0/';
metric = {'ACC','nmi','Purity','Fscore','Precision','Recall','AR','Entropy'};
beta_v = [0.5];

for dsi = 1:length(ds)
    for beta =1:length(beta_v)
    % load data & make folder
    dataName = ds{dsi}; disp(dataName);
    load(strcat(dsPath,dataName));
    
    %TRANS FOR PER DATASETS
    Y = truelabel{1};
    X = data;
    
    k = length(unique(Y));
    
    
    matpath = strcat(resPath,dataName);
    txtpath = strcat(resPath,strcat(dataName,'.txt'));
    if (~exist(matpath,'file'))
        mkdir(matpath);
        addpath(genpath(matpath));
    end
    dlmwrite(txtpath, strcat('Dataset:',cellstr(dataName), '  Date:',datestr(now)),'-append','delimiter','','newline','pc');
    
    %% para setting

     anchor_rate=[1 2 3 4 5 6 7];
     d_rate = [1 2 3 4 5 6 7];

    lambda=0;
    ACC_vector = [];
    NMI_vector = [];
    Purity_vector = [];
    FScore_vector = [];
    %%
    for ichor = 1:length(anchor_rate)
        for id = 1:length(d_rate)
            tic;

              [U,A,W,F,iter,obj,alpha] = algo_qp_test_8(X,Y,lambda,d_rate(id)*k,anchor_rate(ichor)*k,beta_v(beta)); % X,Y,lambda,d,numanchor,proposed method

              res = myNMIACCwithmean(U,Y,k); % [ACC nmi Purity Fscore Precision Recall AR Entropy]
              timer(ichor,id)  = toc;
            
            fprintf('Anchor:%d \t Dimension:%d\t beta:%2.1f\t iter:%d\t Res:%12.6f %12.6f %12.6f %12.6f \tTime:%12.6f \n',[anchor_rate(ichor)*k d_rate(id)*k beta_v(beta) iter res(1) res(2) res(3) res(4) timer(ichor,id)]);
            

            ACC_vector = [ACC_vector res(1)];
            NMI_vector = [NMI_vector res(2)];
            Purity_vector = [Purity_vector res(3)];
            FScore_vector = [FScore_vector res(4)];
            
            resall{ichor,id} = res;
            objall{ichor,id} = obj;
            A_all{ichor,id} =  A;
            U_all{ichor,id} =  U;
            dlmwrite(txtpath, [anchor_rate(ichor)*k d_rate(id)*k 0 res timer(ichor,id)],'-append','delimiter','\t','newline','pc');

        end
    end
%     clear resall objall X Y k
ACC_mean = mean(ACC_vector);ACC_std = std(ACC_vector);[~,Ind_ACC]= max(ACC_vector);
NMI_mean = mean(NMI_vector);NMI_std = std(NMI_vector);[~,Ind_NMI]= max(NMI_vector);
Purity_mean = mean(Purity_vector);Purity_std = std(Purity_vector);[~,Ind_Pur]= max(Purity_vector);
FScore_mean = mean(FScore_vector);FScore_std = std(FScore_vector);[~,Ind_FScore]= max(FScore_vector);
fprintf('mean+-std \t Res:%12.6f+-%7.6f %12.6f+-%7.6f %12.6f+-%7.6f %12.6f+-%7.6f \n', [ACC_mean ACC_std NMI_mean NMI_std Purity_mean Purity_std FScore_mean FScore_std]);
fprintf('best_result_ACC \t Res:%12.6f %12.6f %12.6f %12.6f \n', [max(ACC_vector) NMI_vector(Ind_ACC) Purity_vector(Ind_ACC) FScore_vector(Ind_ACC)]);
fprintf('best_result_NMI \t Res:%12.6f %12.6f %12.6f %12.6f \n', [ACC_vector(Ind_NMI) max(NMI_vector) Purity_vector(Ind_NMI) FScore_vector(Ind_NMI)]);
fprintf('best_result_Purity \t Res:%12.6f %12.6f %12.6f %12.6f \n', [ACC_vector(Ind_Pur) NMI_vector(Ind_Pur) max(Purity_vector) FScore_vector(Ind_Pur)]);
fprintf('best_result_FScore \t Res:%12.6f %12.6f %12.6f %12.6f \n\n', [ACC_vector(Ind_FScore) NMI_vector(Ind_FScore) Purity_vector(Ind_FScore) max(FScore_vector)]);
clear resall objall X Y k
end
end



