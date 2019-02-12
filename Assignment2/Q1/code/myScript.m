
% imtool('image.jpg')

x_i = [1532,1652,1525,2512,2573,2687];
y_i = [1243,1086,931,1221,923,1066];
X_i_whole = [ x_i' y_i' ones(6,1) ]'  ;

delta = 0.1 ;
X_w = [0,0,0   ,3 + delta,4 + delta,6 + delta];
Y_w = [4,5,6   ,4,6,5];
Z_w = [-5 - delta,-4 - delta,-5 - delta,0,0,0];
X_w_whole = [ X_w' Y_w' Z_w' ones(6,1) ]'  ;



%another dataset
% x_i = [692,792,691,1588,1665,1800];
% y_i = [879,1020,1162,1168,892,1035];
% delta = 0.1 ;
% checker_unit = 1;
% X_w = [4*checker_unit+delta,5*checker_unit+delta,6*checker_unit+delta,4*checker_unit+delta,6*checker_unit+delta,5*checker_unit+delta];
% Y_w = [0,0,0,3*checker_unit+delta,4*checker_unit+delta,6*checker_unit+delta];
% Z_w = [5*checker_unit+delta,4*checker_unit+delta,5*checker_unit+delta,0,0,0];
% X_i_whole = [ x_i' y_i' ones(6,1) ]'  ;
% X_w_whole = [ X_w' Y_w' Z_w' ones(6,1) ]'  ;

% another dataset
% X_i_whole=[2487.63, 2482.42 , 1 ;
%           2641.90, 2530.72 , 1 ;
%           2489.95, 2278.28 , 1 ;
%           2301.25, 2404.86 , 1 ;
%           2302.59, 2299.91 , 1 ;
%           2642.32, 2317.71 , 1  ]' ;

% X_w_whole = [
% 0 , 0 , 0 , 1 ;
% 1 , 0 , 0 , 1 ;
% 0 , 1 , 0 , 1 ;
% 0 , 0 , 1 , 1 ;
% 0 , 1 , 1 , 1 ;
% 1 , 1 , 0 , 1  ]'  ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalization

T = eye(3);
U = eye(4);

translation = [ eye(3,2) [ - sum(x_i)/6 ; - sum(y_i)/6 ; 1 ] ] ;
x_i1_transled = translation * X_i_whole;
x_i1_transled_2 = x_i1_transled.^2;
avg_distance = sum(sqrt( x_i1_transled_2(1,:) + x_i1_transled_2(2,:)  ))/6 ;
translation_x  =   [ eye(3,2)  [ 0 ; 0 ;  avg_distance/sqrt(2)  ]   ]  * translation  / (avg_distance/sqrt(2)) ;
x_normalized = translation_x * X_i_whole  ;
T  = translation_x;


translation = [ eye(4,3) [ - sum(X_w)/6 ; - sum(Y_w)/6 ;  - sum(Z_w)/6 ;  1 ] ] ;
X_w1_transled = translation * X_w_whole;
X_w1_transled_2 = X_w1_transled .^2;
avg_distance = sum(sqrt( X_w1_transled_2(1,:) + X_w1_transled_2(2,:) + X_w1_transled_2(3,:)  ))/6 ;
U_translation_X  =   [ eye(4,3)  [ 0 ; 0 ;  0 ;  avg_distance/sqrt(3)  ]   ]  * translation  / (avg_distance/sqrt(3)) ;
X_normalized = U_translation_X * X_w_whole  ;
U = U_translation_X;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Normalised Projection matrix

X_normalized';
x_normalized';

% X_normalized' .* x_normalized(1,:)'
a_xi = [ -X_normalized' zeros(6,4)   X_normalized' .* x_normalized(1,:)' ];
a_yi = [ zeros(6,4)  -X_normalized'  X_normalized' .* x_normalized(2,:)' ];
M = [a_xi ; a_yi ] ;

[~,~,V] = svd(M);
p_cap_norm = reshape(V(:,12),[4,3])' ;



%Projection_ Matrix
P_cap = inv(T) * p_cap_norm * U  ;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

H_inf = P_cap(:,1:3) ;
h = P_cap(:, 4);

inv_H_inf = inv(H_inf);

%camera center
X_o = - inv_H_inf * h  ;

[R_inv K_inv] = qr(inv_H_inf);

%rotation matrix R
R = R_inv'  ;

% intrinsic  matrix K
K = inv(K_inv);
K_33 = K(3,3);
K = K / K_33 ;




P_now = K * R * [eye(3)  -X_o ]  ;
x_guessed = P_now * X_w_whole ;
x_guessed = x_guessed ./ x_guessed(3,:)  ;



diff_ = x_guessed - X_i_whole 
diff_2 = diff_ .^ 2;
RMSE = sqrt(sum(sum(diff_2)) /6 )

