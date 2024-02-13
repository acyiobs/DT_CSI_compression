% tmp_synth = load('result_new_data_3\all_avg_nmse_train_on_synth.mat').all_avg_nmse_train_on_synth;
% tmp_real = load('result_new_data_3\all_avg_nmse_train_on_real.mat').all_avg_nmse_train_on_real;

load('new_result_final\all_avg_nmse_train_on_real.mat');
load('new_result_final\all_avg_nmse_train_on_synth.mat');
load('new_result_final\all_avg_nmse_train_on_O1_synth.mat');
load('result_new_data_2\all_nmse_finetune_noselect.mat');
load('result_new_data_2\all_nmse_finetune_select.mat');
% load('result_new_data_1\all_nmse_combine_noselect.mat');
load('result_new_data_2\all_nmse_combine_select.mat');

% all_avg_nmse_train_on_real = [all_avg_nmse_train_on_real, tmp_real];
% all_avg_nmse_train_on_synth = [all_avg_nmse_train_on_synth, tmp_synth];

% num_data = [1000, 2000, 4000, 8000, 16000, 32000 ];
num_data = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];
figure;
% direct generalization
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_O1_synth(:,1:end-1),1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_synth(:,1:end-1),1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_real(:,1:end-1),1)), '-s');
hold on;
% model refinement
semilogx(num_data, 10*log10(mean(all_nmse_finetune_select,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_nmse_finetune_noselect,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_nmse_combine_select,1)), '-s');
grid on;

xlabel('Number of training/refining data points');
ylabel('NMSE (dB)');
legend('Train on baseline data','Train on DT data', 'Train on target data',...
    'Finetune on selected target data', ...
    'Finetune on random target data', ...
    'Reheasal with selected target data');


tmp = [0.0066736, 0.00516633, 0.00454209, 0.00532391,0.00508295, 0.00508159, 0.00495606, 0.00534524];


