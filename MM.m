function [Z,Sbar,A,Q,labels,converge_Z,converge_F,converge_Z_G] = Train_4(X, cls_num, anchor,alpha,gamma,delta)
% X is a cell data, each cell is a matrix in size of d_v *N,each column is a sample;
% cls_num is the clustering number 
% anchor is the anchor number
% alpha,gamma and delta are the parameters
nV = length(X);
N = size(X{1},2);
t=anchor;
nC=cls_num;
%% ============================ Initialization ============================
for k=1:nV
    Z{k} = zeros(t,N); 
    Q{k} = ones(t,t);
    A{k} = zeros(size(X{k},1),t);
    E{k} = zeros(size(X{k},1),N);
    F{k} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); 
    YY{k} = zeros(size(X{k},1),N); 
end
H = zeros(t,N);
%H(:,1:t) = eye(t);

for i=1:nV+1
  J{i} = zeros(t,N);
  W{i} = zeros(t,N);
end

w = zeros(t*N*nV+1,1);
j = zeros(t*N*nV+1,1);
sX = [t, N, nV+1];
Isconverg = 0;epson = 1e-7;
iter = 0;
mu = 0.0001; max_mu = 10e12; pho_mu = 2;
%0.0001
converge_Z=[];
converge_F=[];
converge_Z_G=[];

while(Isconverg == 0)
%% ============================== Update A^v=============================
A_temp={};
for i = 1 :nV
    A_temp{i}=((YY{i}+mu*X{i}-mu*F{i})*H'*Q{i}'+ (Y{i}+mu*X{i}-mu*E{i})*Z{i}');
    [Au,~,Av] = svd(A_temp{i},'econ');
    A{i}=Au*Av';
end


%% ============================== Update Q^v=============================
Q_temp={};
for i = 1 :nV
    Q_temp{i}=A{i}'*(YY{i}+mu*X{i}-mu*F{i})*H';
    [Qu,~,Qv] = svd(Q_temp{i},'econ');
    Q{i}=Qu*Qv';
end
%% ============================== Update E^v ,F^v=============================
for i=1:nV
  [E{i}] = solve_l1l2(X{i}-A{i}*Z{i}+Y{i}/mu,gamma/mu);
  [F{i}] = solve_l1l2(X{i}-A{i}*Q{i}*H+YY{i}/mu,alpha/mu);
end

  %% =========================== Update Z^k ===========================
for k =1:nV
    tmp = 2*mu*eye(t,t);
    Z_temp = mu*A{k}'*X{k}-mu*A{k}'*E{k}+A{k}'*Y{k}+mu*J{k}-W{k};
    Z{k} = inv(tmp)*Z_temp;
    for ii = 1:size(X{k},2)
        Z{k}(:,ii) = EProjSimplex_new(Z{k}(:,ii));
    end
    %temp_E=[temp_E;(X{k}-A{k}*Z{k}+Y{k}/mu)];
end
%% ============================== Update H=============================
temp_H=0;
for i=1:nV
    H= temp_H+inv(2*mu*eye(t))*(mu*Q{i}'*A{i}'*X{i}+mu*J{nV+1}-W{nV+1}-mu*Q{i}'*A{i}'*F{i}+Q{i}'*A{i}'*YY{i});
    for ii = 1:size(X{i},2)
        H(:,ii) = EProjSimplex_new(H(:,ii));
    end
end
%% ============================= Update J^k ============================== 
t_Z=[Z,{H}];
Z_tensor = cat(3, t_Z{:,:});%%把所有的矩阵堆叠为张量 n*n*V+1
W_tensor = cat(3, W{:,:});
z = Z_tensor(:);
w = W_tensor(:);
J_tensor = solve_G(Z_tensor + 1/mu*W_tensor,mu,sX,delta);
j = J_tensor(:);
%TNN
% [j,objV] = wshrinkObj(Z_tensor + 1/mu*W_tensor,1/mu,sX,0,3);
% J_tensor=reshape(j, sX);
%% ============================== Update W ===============================
w = w + mu*(z - j);
W_tensor = reshape(w, sX);
for k=1:nV+1
    W{k} = W_tensor(:,:,k);
end
%% ============================== Update Y YY ===============================

for i=1:nV
    Y{i} = Y{i} + mu*(X{i}-A{i}*Z{i}-E{i});
    YY{i} = YY{i} + mu*(X{i}-A{i}*Q{i}*H-F{i});
end
%% ====================== Checking Coverge Condition ======================
    max_Z=0;
    max_Z_F=0;
    max_Z_G=0;
    max_H_G=0;
    Isconverg = 1;
    for k=1:nV
        if (norm(X{k}-A{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-A{k}*Z{k}-E{k},inf);
            Isconverg = 0;
            max_Z=max(max_Z,history.norm_Z);
        end
        if (norm(X{k}-A{k}*Q{k}*H-F{k},inf)>epson)
            history.norm_ZF = norm(X{k}-A{k}*Q{k}*H-F{k},inf);
            Isconverg = 0;
            max_Z_F=max(max_Z_F,history.norm_ZF);
        end
        for i=1:nV+1
            J{i} = J_tensor(:,:,i);
            W_tensor = reshape(w, sX);
            W{i} = W_tensor(:,:,i);
        end
        for k=1:nV+1
            if (norm(t_Z{k}-J{k},inf)>epson)
                history.norm_Z_G = norm(t_Z{k}-J{k},inf);
                Isconverg = 0;
                max_Z_G=max(max_Z_G, history.norm_Z_G);
            end
        end
    end
    converge_Z=[converge_Z max_Z];
    converge_F=[converge_F max_Z_F];
    converge_Z_G=[converge_Z_G max_Z_G];
   
    
    if (iter>19)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
end

Sbar=[];
for i = 1:nV+1
    Sbar=cat(1,Sbar,1/sqrt(nV+1)*(t_Z{i}));
end
%Sbar=H;

[U,Sig,V] = mySVD(Sbar',nC); 


rand('twister',5489)
labels=litekmeans(U, nC, 'MaxIter', 100,'Replicates',10);%kmeans(U, c, 'emptyaction', 'singleton', 'replicates', 100, 'display', 'off');
end
