function r = data_rate(p,h)
% calculate the data rates
% input--UENum:number of users
%        ChannelNum:number of subcarrier
%        W: bandwidth
%        p: power allocation
%        h: channel gain
%        N0: spectrum density of AGWN 
%        rho: assignment of subcarrier
% output--data rates of each user
N0PSD = -174; % noise power spectrum density, dBm/Hz
N0 = 10.^((N0PSD-30)/10)*1*10^6;

r_UtoC = log(1 + p.*h/N0)/log(2);
r = sum(r_UtoC,2);

end