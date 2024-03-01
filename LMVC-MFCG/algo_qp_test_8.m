function [UU,A,W,F,iter,obj,alpha] = algo_qp_test_8(X,Y,alpha,d,numanchor,beta)
% m      : the number of anchor. the size of Z is m*n.
% lambda : the hyper-parameter of regularization term.

% X      : n*di

%% initialize
maxIter = 100 ; % the number of iterations

m = numanchor;
numclass = length(unique(Y));
numview = length(X);
numsample = size(Y,1);

W = cell(numview,1);            % di * d
A = zeros(d,m);         % d  * m
Z = zeros(m,numsample); % m  * n

for i = 1:numview
   X{i} = mapstd(X{i},0,1); % turn into d*n 
   di = size(X{i},1); 
   W{i} = zeros(di,d);
end
Z(:,1:m) = eye(m);

%Initilize G,F
G = eye(m,numclass);
F = ones(numclass,numsample)/numclass; 

alpha = ones(1,numview)/numview;

opt.disp = 0;
 
flag = 1;
iter = 0;

%%
while flag
    iter = iter + 1;
    
    %% optimize W_i
    parfor iv=1:numview
        C = X{iv}*F'*G'*A';      
        [U,~,V] = svd(C,'econ');
        W{iv} = U*V';
    end

    %% optimize A2
    sumAlpha = 0;
    part1 = 0;

       for ia = 1:numview
        al2 = alpha(ia)^2;
        sumAlpha = sumAlpha + al2;
        part1 = part1 + al2 * W{ia}' * X{ia} * F'* G';
       end

    [Unew,~,Vnew] = svd(part1,'econ');
    A = Unew*Vnew';

    %% optimize G
    sumAlpha_G = 0;
    part1_G = 0;
    for ia = 1:numview
        al2_G = alpha(ia)^2;
        sumAlpha_G = sumAlpha_G + al2_G;
        part1_G = part1_G + al2_G * A' * W{ia}' * X{ia} * F';
    end
    [Unew_G,~,Vnew_G] = svd(part1_G,'econ');
    G = Unew_G*Vnew_G';
    
    %% optimize F
    options = optimset( 'Algorithm','interior-point-convex','Display','off');
    G_F = zeros(numclass);
    for a=1:numview
        G_F = G_F + 2*(alpha(a)^2+beta)*eye(numclass); 
    end
    G_F = (G_F+G_F')/2;
    

    
    parfor ji=1:numsample
        f_F = zeros(1,numclass);
        for j=1:numview
            B_F=X{j};
            f_F= f_F + (-2*B_F(:,ji)'* W{j}*A *G);
        end
        F(:,ji)=quadprog(G_F,f_F,[],[],ones(1,numclass),1,zeros(numclass,1),ones(numclass,1),[],options);
    end
    

    %% optimize alpha
    M = zeros(numview,1);
    for iv = 1:numview
        M(iv) = norm( X{iv} - W{iv} * A * G * F,'fro')^2;
    end
    Mfra = M.^-1;
    Q = 1/sum(Mfra);
    alpha = Q*Mfra;

    %%
    term1 = 0;
    for iv = 1:numview
        term1 = term1 + alpha(iv)^2 * norm(X{iv} - W{iv} * A * G * F,'fro')^2;
    end

    obj(iter) = term1;

    
      if (iter>1) && (abs((obj(iter-1)-obj(iter))/(obj(iter-1)))<1e-4 || iter>maxIter || obj(iter) < 1e-10)
        UU = F';
        UU = UU(:,1:numclass);
        flag = 0;
    end
end
         
         
    
