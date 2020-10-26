function [c,ceq] = rate(x)

load const.mat
n_d =length(rho_d);
R = zeros(L,1);


for l = 1:L
    for k = 1:K
        if (k-1)*L+l<=n_d
            R(l) =  R(l) + log(1+rho_d((k-1)*L+l)*x(1+(k-1)*L+l+K*L-n_d)/(a(k,l)*rho_d((k-1)*L+l) + b(k,l)*x(1+(k-1)*L+l+K*L-n_d)))/log(2);
        else
            R(l) = R(l) + log(1+x(1+(k-1)*L+l-n_d)*x(1+(k-1)*L+l+K*L-n_d)/(a(k,l)*x(1+(k-1)*L+l-n_d) + b(k,l)*x(1+(k-1)*L+l+K*L-n_d)))/log(2);
        end
    end
end
c = ones(L,1)*x(1)-R;
ceq = [];

end