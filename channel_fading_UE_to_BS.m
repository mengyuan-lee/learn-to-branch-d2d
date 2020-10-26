function CUEPL=channel_fading_UE_to_BS(Distance_km)
% distance: km
std=10;
CUEPL=128.1+37.6*log10(Distance_km)+normrnd(0,std);
