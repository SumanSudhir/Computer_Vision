
function [ H ] = homography( p1, p2 )

x1 = p1(1,1);
x2 = p1(2,1);
x3 = p1(3,1);
x4 = p1(4,1);

y1 = p1(1,2);
y2 = p1(2,2);
y3 = p1(3,2);
y4 = p1(4,2);

x1_desh = p2(1,1);
x2_desh = p2(2,1);
x3_desh = p2(3,1);
x4_desh = p2(4,1);

y1_desh = p2(1,2);
y2_desh = p2(2,2);
y3_desh = p2(3,2);
y4_desh = p2(4,2);


P = [-x1 -y1 -1 0 0 0 x1*x1_desh y1*x1_desh x1_desh;
    0 0 0 -x1 -y1 -1 x1*y1_desh y1*y1_desh y1_desh;
    
    -x2 -y2 -1 0 0 0 x2*x2_desh y2*x2_desh x2_desh;
    0 0 0 -x2 -y2 -1 x2*y2_desh y2*y2_desh y2_desh;
    
    -x3 -y3 -1 0 0 0 x3*x3_desh y3*x3_desh x3_desh;
    0 0 0 -x3 -y3 -1 x3*y3_desh y3*y3_desh y3_desh;
    
    -x4 -y4 -1 0 0 0 x4*x4_desh y4*x4_desh x4_desh;
    0 0 0 -x4 -y4 -1 x4*y4_desh y4*y4_desh y4_desh];


[U,S,V] = svd(P);
h = V(:,9)';
H = [h(1:3);h(4:6);h(7:9)];

end

