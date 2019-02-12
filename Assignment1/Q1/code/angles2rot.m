function rot_matrix = angles2rot(angles_list)
    %% Your code here
    % angles_list: [theta1, theta2, theta3] about the x,y and z axes,
    % respectively.
    [length_1,length_2]=size(angles_list);
    rot_matrix=zeros(length_1,3,3);
    
    for i = 1:length_1
        wx=angles_list(i,1);
        wy=angles_list(i,2);
        wz=angles_list(i,3);
        Rx= [1,0,0;     0,cosd(wx),-sind(wx);      0,sind(wx),cosd(wx)];
        Ry= [cosd(wy),0,sind(wy);      0,1,0;     -sind(wy),0,cosd(wy)];
        Rz= [cosd(wz),-sind(wz),0;     sind(wz),cosd(wz),0;      0,0,1];
        %final matrix for each bone is calculated by first rotating around x, then y then z axis;
        rot_matrix(i,:,:) = Rz * Ry * Rx;

    end

end




