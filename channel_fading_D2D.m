function CUEPL=channel_fading_D2D(Distance_km)
%Distance: km
std=10;
CUEPL=148+40*log10(Distance_km)+normrnd(0,std);
