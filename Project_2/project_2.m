close all
clear all
clc

table = readtable('dat_intel.csv');
data = table.Close;

returns = computeReturns(data);
log_returns = computeLogReturns(data);

plot(log_returns)
title('Log Returns')

autocorr(log_returns, 50)

%% Functions

function log_ret = computeLogReturns(data)
X_tp1 = data(2 : end);
X = data(1 : end-1);
log_ret = log(X_tp1) - log(X);
end

function ret = computeReturns(data)
X_tp1 = data(2 : end);
X = data(1 : end-1);
ret = X_tp1 - X;
end