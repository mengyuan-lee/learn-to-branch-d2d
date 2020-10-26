function [] = main_generate(K,L,num,train_flag)
R_min_C = 2; %bit/s/hz
P_max_D = 0.1; %w
P_max_C = 0.1; %w

H_CB = [];
H_DB = [];
H_CD = [];
H_D = [];

R1 = 500; R2 = 250; %m 250-500
r1 = 50;  r2 = 15;%m 15-50

i=1;
while i<=num
    [CUx,CUy] = create_random_location(R1,R2,K,0,0);
    [DTx,DTy] = create_random_location(R1,R2,L,0,0);
    for l = 1:L
        [DRx(l),DRy(l)]= create_random_location(r1,r2,1,DTx(l),DTy(l));
    end
    h_CB = Fun_chGain(K,CUx,CUy);
    h_DB = Fun_chGain(L,DTx,DTy);
    h_CD = Fun_chGainD2D(K,L,CUx,CUy,DRx,DRy);
    h_D = diag(Fun_chGainD2D(L,L,DTx,DTy,DRx,DRy));

    rho_d = [ ];
    [rho,p,yita_max,exitflag] = minlp_solve(K,L,R_min_C,P_max_D,P_max_C,h_CD,h_D,h_CB,h_DB,rho_d);
    if exitflag == 1
        H_CB = [H_CB;h_CB'];
        H_DB = [H_DB;h_DB'];
        H_D = [H_D;h_D'];
        temp1 = h_CD';
        temp2 = temp1(:);
        H_CD = [H_CD;temp2'];
        i = i+1;
    end
    
end
filename = ['data_',mat2str(K),'_',mat2str(L),'_',train_flag,'.mat'];
save(filename)
end
