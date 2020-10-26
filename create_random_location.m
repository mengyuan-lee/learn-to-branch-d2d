function [x,y] = create_random_location(R1,R2,num,cx,cy)
% create random location for D2D pairs
%--------------------------------------------%
x = zeros(num,1);
y = zeros(num,1);
i = 1;
while num>=i    
    x(i,1) = 2*R1*rand-R1+cx; 
    y(i,1) = 2*R1*rand-R1+cy;
    if (x(i)-cx)^2+(y(i)-cy)^2 < R1^2 
        if (x(i)-cx)^2+(y(i)-cy)^2 > R2^2
        i=i+1;
        end
    end
end