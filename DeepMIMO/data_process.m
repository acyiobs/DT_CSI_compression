% clear;
% clc;
% load('DeepMIMO\DeepMIMO_dataset\dataset4\DeepMIMO_dataset.mat');

num_user = numel(DeepMIMO_dataset{1, 1}.user);
all_LoS = zeros(num_user,1);


num_user = numel(DeepMIMO_dataset{1, 1}.user);
channel_shape = size(DeepMIMO_dataset{1, 1}.user{1,1}.channel);
all_channel = zeros([num_user, channel_shape]);
all_pos = zeros([num_user, 3]);
all_LoS = zeros(num_user, 1);

%% extract channels and positions
for u=1:num_user
    user_channel = DeepMIMO_dataset{1, 1}.user{1, u}.channel; % (rx, tx, subcarrier)
    user_pos = DeepMIMO_dataset{1, 1}.user{1, u}.loc;
    user_LoS = DeepMIMO_dataset{1, 1}.user{1, u}.LoS_status;
    all_channel(u, :, :, :, :) = single(user_channel);
    all_pos(u, :) = user_pos;
    all_LoS(u) = user_LoS;
end
all_channel = single(all_channel);
all_pos = single(all_pos);
all_LoS = single(all_LoS);

user_with_path = (all_LoS~=-1);
all_channel = all_channel(user_with_path, :, :, :, :);
all_pos = all_pos(user_with_path, :);
all_LoS = all_LoS(user_with_path);





