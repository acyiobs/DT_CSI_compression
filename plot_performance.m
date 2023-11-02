load('result_new_data_2\all_avg_nmse_train_on_real.mat');
load('result_new_data_2\all_avg_nmse_train_on_synth.mat');
load('result_new_data_1\all_nmse_finetune_noselect.mat');
load('result_new_data_1\all_nmse_finetune_select.mat');
load('result_new_data_1\all_nmse_combine_noselect.mat');
load('result_new_data_1\all_nmse_combine_select.mat');

% num_data = [1000, 2000, 4000, 8000, 16000, 32000 ];
num_data = [10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120];
figure;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_real,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_avg_nmse_train_on_synth,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_nmse_finetune_noselect,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_nmse_finetune_select,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_nmse_combine_noselect,1)), '-s');
hold on;
semilogx(num_data, 10*log10(mean(all_nmse_combine_select,1)), '-s');
grid on;
xlabel('Number of training data points');
ylabel('NMSE (dB)');
legend('Train on real', 'Train on synth', ...
    'Finetune on random real (pretrain on 5k synth data points)', ...
    'Finetune on selected real (pretrain on 5k synth data points)', ...
    'Augment with random real (pretrain on 5k synth data points)', ...
    'Augment with selected real (pretrain on 5k synth data points)');

