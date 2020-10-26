function R_D = R_rate(a,b,rho,s,K,L)
R_D = zeros(L,1);
for l = 1:L
    for k = 1:K
        R_D(l) = R_D(l) + log(1+rho((k-1)*L+l)*s((k-1)*L+l)/(a(k,l)*rho((k-1)*L+l) + b(k,l)*s((k-1)*L+l)))/log(2);
    end
end

end
