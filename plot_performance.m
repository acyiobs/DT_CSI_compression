load('result_new_data_1\all_avg_nmse_train_on_real.mat');
load('result_new_data_1\all_avg_nmse_train_on_synth.mat');
% load('result4\all_avg_nmse_finetune.mat');
% load('result3\all_nmse_finetune_select_combine.mat');
% num_data = [1000, 2000, 4000, 8000, 16000, 32000 ];
num_data = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];
figure;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_real,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_synth,1)), '-s');
hold on;
% semilogx(num_data, 10*log10(mean(all_avg_nmse_finetune,1)), '-s');
% hold on;
% semilogx(num_data, 10*log10(all_nmse_finetune_select_combine), '-s');
grid on;
xlabel('Number of training data points');
ylabel('NMSE (dB)');
legend('Train on real', 'Train on synth', 'Finetune on real (pretrain on 32k synth data points)', 'Finetune on selected real (pretrain on 32k synth data points)');

