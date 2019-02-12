%% MyMainScript
% ransacHomography function done
tic;
%% Your code here
i1=imread('../input/hill/1.JPG');
i2=imread('../input/hill/2.JPG');
i3=imread('../input/hill/3.JPG');
[p1,da]=vl_sift(single(rgb2gray(i1)));
[p2,db]=vl_sift(single(rgb2gray(i2)));
[matches,~]=vl_ubcmatch(da,db,90);
[~,y_matches]=size(matches);
r_1=randsample(y_matches,4);
%h=ransacHomography(p1,p2,45);


toc;
