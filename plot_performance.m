load('new_result_final\all_avg_nmse_train_on_high_res_synth.mat');
all_avg_nmse_train_on_high_res_synth = all_avg_nmse_train_on_synth;
load('new_result_final\all_avg_nmse_train_on_low_res_synth.mat');
all_avg_nmse_train_on_low_res_synth = all_avg_nmse_train_on_synth;
load('new_result_final\all_avg_nmse_train_on_mid_res_synth.mat');
all_avg_nmse_train_on_mid_res_synth = all_avg_nmse_train_on_synth;

load('new_result_final\all_avg_nmse_train_on_real.mat');
load('new_result_final\all_avg_nmse_train_on_synth.mat');
load('new_result_final\all_avg_nmse_train_on_O1_synth.mat');
load('new_result_final\all_nmse_finetune_noselect_.mat');
load('new_result_final\all_nmse_finetune_select_.mat');
% load('result_new_data_1\all_nmse_combine_noselect.mat');
load('new_result_final\all_nmse_combine_select_.mat');

num_data = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];
figure;
% direct generalization
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_O1_synth(:,1:10),1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_low_res_synth,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_mid_res_synth,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_high_res_synth,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_synth(:,1:10),1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_real(:,1:10),1)), '-s');

%plot(num_data,zeros(1,10)-17.0337, '--')
grid on;

xlabel('Number of training data points');
ylabel('NMSE (dB)');
legend('Train on baseline data', ...
    'Train on low-res DT data', ...
    'Train on mid-res DT data', ...
    'Train on high-res DT data', ...
    'Train on perfect DT data', ...
    'Train on target data');

