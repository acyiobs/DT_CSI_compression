load('result3\all_avg_nmse_train_on_real.mat');
load('result3\all_avg_nmse_train_on_synth.mat');
load('result3\all_nmse_finetune.mat');
num_data = [1000, 2000, 4000, 8000, 16000, 32000 ];
figure;
semilogx(num_data, 10*log10(all_avg_nmse_train_on_real), '-s');
hold on;
semilogx(num_data, 10*log10(all_avg_nmse_train_on_synth), '-s');
hold on;
semilogx(num_data, 10*log10(all_nmse_finetune), '-s');
grid on;
xlabel('Number of training data points');
ylabel('NMSE (dB)');
legend('Train on real', 'Train on synth', 'Finetune on real (pretrain on 32k synth data points)');

