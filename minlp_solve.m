function [rho,p,yita_max,exitflag] = minlp_solve(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho_d)
% output: rho/p
%         yita_max: yita
%         exitflag: Reason fmincon stopped, returned as an integer.
% input:  K number of CUs
%         L number of D2D links
%         R_min_C: minimum rate of CUs
%         P_max_D: maximum d2d linnk power
%         P_max_C: maximum CUs
%         h_CD:channel gain between CU and the receiver of D2D pair 
%         h_D:channel gain of D2D pair 
%         h_CB:channel gain between CU and base station
%         h_CB:channel gain between the transmitter of D2D pair and base station
rho_d_k = ceil(length(rho_d)/L);


%rho_d_l = (length(rho_d)-(rho_d_k-1)*L)*(rho_d_k~=0);
if rho_d_k~=0
    rho_d_l = length(rho_d)-(rho_d_k-1)*L;
else
    rho_d_l = 0;
end
    

[K,L,~,~,P_max_D,p_max,rho_d,rho_d_k,rho_d_l]=const(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho_d,rho_d_k,rho_d_l);

%-----------------check feasibility----------------------
%flag = (rho_d_l~=L);
if rho_d_l~=L
    flag = 1;
else
    flag = 0;
end

fflag = 1;
for k = 1:rho_d_k-flag
    sum=0;
    for l = 1:L
        sum = sum + rho_d((k-1)*L+l);
    end
    if sum>1
        fflag = 0;
        break;
    end
end
if flag==1
    k = rho_d_k;
    sum=0;
    for l = 1:rho_d_l
        sum = sum + rho_d((k-1)*L+l);
    end
    if sum>1
        fflag = 0;
    end
end
if fflag == 0
    rho = [];
    p = [];
    yita_max = -1e-6;
    exitflag = -2;
    return;
end
%-----------------check feasibility end----------------------

% objective function
fun = @(x)-x(1);
% initialization
n_d =length(rho_d);
x0 = zeros(2*K*L+1-n_d,1);

%-----------------constraints---------------------
% lower bound and upper bound
lb = zeros(size(x0));
ub = ones(size(x0));
ub(1) = 100;
p_max_t = p_max';
ub(2+K*L-n_d:1+2*K*L-n_d) = p_max_t(:);

% linear inequality 
n_v = length(x0);
A = zeros(K+L+K*L-rho_d_k+flag,n_v);
B = zeros(K+L+K*L-rho_d_k+flag,1);
if flag==1
    for i = 1:L-rho_d_l
        A(1,i+1)=1;
    end
    B(1)=1-sum;
    for k = 1:K-rho_d_k
       for l = 1:L
           A(k+1,(k+rho_d_k-1)*L+l+1-n_d)=1;
       end
       B(k+1)=1;
   end
else
    for k = 1:K-rho_d_k
        for l = 1:L
           A(k,(k-1+rho_d_k)*L+l+1-n_d)=1;
       end
       B(k)=1;
   end
end

for l = 1:L
    for k = 1:K
        A(K+l-rho_d_k+flag,(k-1)*L+l+1+K*L-n_d)=1;
    end
    B(K+l-rho_d_k+flag)=P_max_D;
end

for k = 1:K
    for l = 1:L
        if (k-1)*L+l<=n_d
            A(K+L+(k-1)*L+l-rho_d_k+flag,(k-1)*L+l+1+K*L-n_d)=1;
            B(K+L+(k-1)*L+l-rho_d_k+flag)=rho_d((k-1)*L+l)*p_max(k,l);
        else
            A(K+L+(k-1)*L+l-rho_d_k+flag,(k-1)*L+l+1-n_d)=-p_max(k,l);
            A(K+L+(k-1)*L+l-rho_d_k+flag,(k-1)*L+l+1+K*L-n_d)=1;
            B(K+L+(k-1)*L+l-rho_d_k+flag)=0;
        end
    end
end

% nolinear constraints

nonlcon = @rate;
%----------------constraints end---------------------
Aeq = [];
Beq = [];
% solve problem
options = optimoptions('fmincon','MaxIterations',1e5,'MaxFunctionEvaluations',7e5,'TolFun',1e-3);
[x,fval,exitflag] = fmincon(fun,x0,A,B,Aeq,Beq,lb,ub,nonlcon,options);
rho = [rho_d';x(2:K*L-n_d+1)];
s = x(K*L-n_d+2:length(x0));
p = rho;
for i = 1:length(rho)
    if rho(i)==0
        p(i)=0;
    else
        p(i)=s(i)/rho(i);
    end
end

yita_max = -fval;

end
