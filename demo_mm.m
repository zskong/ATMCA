clear all
clc
addpath('dataset\');
addpath('funs\');
addpath("results\");
dataname=["NGs"];

%% ==================== Load Datatset and Normalization ===================
for it_name = 1:length(dataname)
    load(strcat('dataset/',dataname(it_name),'.mat'));
    cls_num=length(unique(Y));
    gt = Y;
    nV = length(X);
    k=max(Y);
    temp=X;
    %% original structure information
    for v=1:nV
        [X{v}]=NormalizeData(X{v}');
    end
    %% ========================== Parameters Setting ==========================
    result=[];
    num = 0;
    max_val=0;
    record_num = 0;
    ii=1;
    %% ============================ Optimization =============================
    alpha = 10^(-6);
    gamma = 10^(-1);
    delta=  10^(0);
    anchor = 2*cls_num;
    tic;
    [Z,Sbar,A,Q,y,converge_Z,converge_F,converge_Z_G] = MM(X, cls_num, anchor,alpha,gamma,delta);
    time = toc;
    %[ii,jj,hh,kk]=findnumber(1600,-6,1,-6,1,-6,1,1,7)%%查找具体某个数据对应的参数
    [result(ii,:)]=  Clustering8Measure(gt, y);
    fprintf('\n alpha:%.6f gamma:%.6f delta:%.6f anchor:%.1f\n ACC: %.4f NMI: %.4f ARI: %.4f Time: %.4f \n',[alpha gamma delta anchor result(ii,1) result(ii,2) result(ii,7) time]);
end