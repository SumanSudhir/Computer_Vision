function [result_pose, composed_rot] = transformPose(rotations, pose, kinematic_chain, root_location)
    % rotations: A 15 x 3 x 3 array for rotation matrices of 15 bones
    % pose: The base pose coordinates 16 x 3.
    % kinematic chain: A 15 x 2 array of joint ordering
    % root_positoin: the index of the root in pose vector.
    % Your code here 

    %calculated(i) gives 1 if result_pose of joint is calculated else 0;
    completed = zeros(1,16);
    completed(1,root_location) = 1;

    %result pose for each joint
    result_pose = zeros(16,3);
    result_pose(root_location,:)=pose(root_location,:);
    
    composed_rot = zeros(size(rotations));

    %it gives the index of bone(coming from root) for a joint
    parent_node_bone_index = zeros(16,0);
    
    for i=1:15
        parent_node_bone_index(kinematic_chain(i,1),1) = i;
    end

    while sum(completed) < 15
        %for each bone if parent's composed_rot and pos is calculated
        for i=1:15
            if  completed(1,kinematic_chain(i,1)) < 1  && completed(1,kinematic_chain(i,2)) > 0
                %if second end of bone is root then the total rotation is only because of bone rotation else it will be also get some extra rotation what their parental bone does.
                if kinematic_chain(i,2) == root_location
                    composed_rot(i,:,:) = reshape(rotations(i,:,:),[3,3]);
                else 
                    composed_rot(i,:,:) =   reshape(composed_rot(parent_node_bone_index(kinematic_chain(i,2),1),:,:),[3,3]) * reshape(rotations(i,:,:),[3,3]);
                end
                
                %bone vector
                pose_diff = pose(kinematic_chain(i,1),:) - pose(kinematic_chain(i,2),:);
                matrix_here = reshape(composed_rot(i,:),[3,3]);
                %final bone vector for the bone
                to_be_added = (matrix_here * pose_diff.').';
                
                result_pose(kinematic_chain(i,1),:) = result_pose(kinematic_chain(i,2),:) + to_be_added ;
                completed(1,kinematic_chain(i,1)) = 1;
            end
        end
    end
end