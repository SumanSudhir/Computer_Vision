%% Rigit Transform between 2 sets of 3D Points

%% Load Data
p1 = [847 682;1060 719; 1126 556; 962 536];
p2 = [0 0;18 0;18 44;0 44];


H = homography(p1,p2);
X = H*([1023,812,1]');
X = X./X(3);

%% Your code here