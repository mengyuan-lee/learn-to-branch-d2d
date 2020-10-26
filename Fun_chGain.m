function H = Fun_chGain(ChannelNum,CUE_x,CUE_y)
%input: ChannelNum : number of channel
%       CUE_x: abcissa of users
%       CUE_y; ordinate of users
% output:H: Channel Gain

% comppute distance 
% regard cloud as the geometic center
Dis = zeros(ChannelNum,1);
for  i = 1:ChannelNum
            Dis(i)= sqrt(CUE_x(i)^2 + CUE_y(i)^2);
end

%the channel gain of user i
H = zeros(ChannelNum,1);
for k = 1:ChannelNum
        H(k) = channel_fading_UE_to_BS(Dis(k)/1000);
        H(k) = 10.^(-H(k)/10);
end

end