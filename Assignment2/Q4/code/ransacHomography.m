function [ H ] = ransacHomography( x1, x2, thresh )
%   RANSACHOMOGRAPHY Summary of this function goes here
%   From two sets of points x1 and x2,it selects randomly some 'chose' 
%   number of indices and finds the homography matrix with the points
%   in the set x1 and x2 cooresponding to these indices

%   Detailed explanation goes here
%   The points in p1 and p2 correspond to same indices in x1 and
%   x2,repectively.'g_h' is the homography matrix obtained from the point
%   sets p1 and p2.'max_score' stores the maximum score corresponding
%   to a homography matrix so far.
[x,~]=size(x1);
chose=4;
max_score=-1;
for i=1:20
    k=randsample(x,chose);
    p1=x1(k);
    p2=x2(k);
    g_h=homography(p1,p2);
    p2=g_h*p1;
    score=0;
    for j=1:x
        eror=sqrt((p2(j,1)-p1(j,1))^2+(p2(j,2)-p1(j,2))^2);
        if eror < thresh
            score=score+1;
        end
    end
    if max_score<score
        max_score=score;
        H=g_h;
    end
end
end