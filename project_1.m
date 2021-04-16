clc
clear all
close all

table = readtable('intel.csv');
data = table.Volume;
data_missing = table.VolumeMissing;

n = length(data);

X_tp1    = data(1 : end - 1);
X        = data(2 : end);

X_missing_tp1    = data_missing(1 : end - 1);
X_missing        = data_missing(2 : end);

returns         = X_tp1 - X;
log_returns     = log(X_tp1) - log(X);

missing_returns         = X_missing_tp1 - X_missing;
log_missing_returns     = log(X_missing_tp1) - log(X_missing);

corr_returns            = returns - mean(returns);
corr_missing_returns    = missing_returns - mean(missing_returns);

corr_log_returns            = log_returns - mean(log_returns);
corr_log_missing_returns    = log_missing_returns - mean(log_missing_returns);

[acf, lags, bounds, h] = autocorr(log_returns);

figure;
plot(log_returns)
title('Log Returns')

figure;
stem(acf)
title('Autocorrelation')