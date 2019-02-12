%% When aligning barbara and negative barbara,Translation_x=3 and Theta=-24 
%% When aligning flash1 and noflash1,Translation_x=3 and Theta=-23
f_i=imread('../input/barbara.png');
rm_i=imread('../input/negative_barbara.png');
[~,~,z]=size(f_i);
%downsize
if z==3
    f_i=imresize(f_i,1/3);
    f_i=rgb2gray(f_i);
    rm_i=imresize(rm_i,1/3);
    rm_i=rgb2gray(rm_i);
end
[xfi,yfi]=size(f_i);
xb2=floor(xfi/2);
yb2=floor(yfi/2);
xby10=ceil(xfi/10);
yby10=ceil(yfi/10);

r_tx=-3; r_theta=23.5;

m_i=imrotate(rm_i,r_theta);

m_i=imtranslate(m_i,[r_tx,0],'OutputView','full');
%add noise
rm_i=double(rm_i)+16*rand(xfi,yfi)-8;
rm_i=uint8(rm_i);
% guess_theta and guess_tx give the theta and t_x with which the two images being
% aligned have the minimum entropy
guess_tx=-12;
guess_theta=-60;
min_entrp=676;
j_entrp=zeros(25,121);
for tx=-12:12
    for theta=-60:60
        rot=theta+r_theta;
        trnslate=tx+r_tx;
        t_mi=imrotate(rm_i,rot);
        [x,y]=size(t_mi);
        x=floor(x/2);
        y=floor(y/2);
        t_mi=t_mi(x-xb2+1:x+xb2,:);
        t_mi=imtranslate(t_mi,[trnslate,0]);
        t_mi=t_mi(:,y-yb2+1:y+yb2); 
        j_hist=zeros(26,26);
        for i=1:xfi
            for j=1:yfi
                h_fi=floor(double(f_i(i,j))/10)+1;
                h_tmi=floor(double(t_mi(i,j))/10)+1;
                j_hist(h_fi,h_tmi)=j_hist(h_fi,h_tmi)+1;
            end
        end
        j_hist=j_hist/676;
        current_i=13+tx;
        current_j=61+theta;
        for i=1:26
            for j=1:26
                if j_hist(i,j)~=0
                j_entrp(current_i,current_j)=j_entrp(current_i,current_j)+j_hist(i,j)*log2(j_hist(i,j));
                end
            end
        end
        j_entrp(current_i,current_j)=-j_entrp(current_i,current_j);
        if j_entrp(current_i,current_j)<min_entrp
            min_entrp=j_entrp(current_i,current_j);
            guess_tx=tx;
            guess_theta=theta;
        end
    end
end
[x,y]=meshgrid(-60:1:60,-12:1:12);
surf(x,y,j_entrp)