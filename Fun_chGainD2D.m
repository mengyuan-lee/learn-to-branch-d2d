function H = Fun_chGainD2D(num1,num2,CU1_x,CU1_y,CU2_x,CU2_y)
% input: num1/num2 :numbers of the first and second sets of users
%       CU1_x,CU1_y,CU2_x,CU2_y: coordinates of the first and second sets of users
% output:H: Channel Gain

% comppute distance 
% regard cloud as the geometic center
Dis = zeros(num1,num2);
for k = 1:num1
    for l = 1:num2
        Dis(k,l)= sqrt((CU1_x(k)-CU2_x(l))^2 + (CU1_y(k)-CU2_y(l))^2);
    end
end

%the channel gain 
H = zeros(num1,num2);
for k = 1:num1
    for l = 1:num2
        H(k,l) = channel_fading_D2D(Dis(k,l)/1000);
        H(k,l) = 10.^(-H(k,l)/10);
    end
end

end
