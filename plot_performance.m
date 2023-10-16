load('result\all_avg_nmse_train_on_real.mat');
load('result\all_avg_nmse_train_on_synth_.mat');
load('result\all_nmse_finetune.mat');
num_data = [1000, 5000, 10000, 50000, 80000];
figure;
plot(num_data, 10*log10(all_avg_nmse_train_on_real), '-s');
hold on;
plot(num_data, 10*log10(all_avg_nmse_train_on_synth), '-s');
hold on;
plot(num_data, 10*log10(all_nmse_finetune), '-s');
grid on;
xlabel('Number of training/fine-tuning data points');
ylabel('NMSE (dB)');
legend('Train on real', 'Train on synth', 'Finetune on real (pretrain on 80k synth data points)');

