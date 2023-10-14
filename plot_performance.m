load('result\all_avg_nmse_train_on_real.mat');
num_data = [1000, 5000, 10000, 50000, 80000];
figure;
plot(num_data, 10*log10(all_avg_nmse_train_on_real), '-s');
grid on;
xlabel('Number of training data points');
ylabel('NMSE (dB)');
