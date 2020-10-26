function [a,b,p_max] = para(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB)
%output:
%        a/b:K*L matrix
%        P_max_D: maximum d2d linnk power
%        p_max: K*L matrix
%input:  K number of CUs
%        L number of D2D links
%        R_min_C: minimum rate of CUs
%        P_max_D: maximum d2d linnk power
%        P_max_C: maximum CUs
%        h_CD:channel gain between CU and the receiver of D2D pair 
%        h_D:channel gain of D2D pair 
%        h_CB:channel gain between CU and base station
%        h_CB:channel gain between the transmitter of D2D pair and base station

N0PSD = -174; % noise power spectrum density, dBm/Hz
N0 = 10.^((N0PSD-30)/10)*1*10^6;

a = zeros(K,L);
for k= 1:K
    for l= 1:L
        a(k,l) = N0/h_D(l)+((2^R_min_C-1)*h_CD(k,l)*N0)/(h_D(l)*h_CB(k));
    end
end

b = zeros(K,L);
for k= 1:K
    for l= 1:L
        b(k,l) = ((2^R_min_C-1)*h_CD(k,l)*h_DB(l))/(h_D(l)*h_CB(k));
    end
end

p_max = zeros(K,L);
for k= 1:K
    for l= 1:L
        p_max(k,l) = (1/h_DB(l))*(P_max_C*h_CB(k)/(2^R_min_C-1)-N0);
        if P_max_D<p_max(k,l)
            p_max(k,l) = P_max_D;
        end
    end
end

end