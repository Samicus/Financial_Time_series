%% Problem 2

close all
clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;

returns = computeReturns(data);
log_returns = computeLogReturns(data);

rho = autocorr(log_returns);
autocorr(log_returns)
%%
q = 4;

gamma = autocov(data, log_returns);
gamma = gamma(1:q);

figure;
plot(log_returns)
ylabel('Log(X_{t+1} - X_t)')
xlabel('t')
title('Log Returns')

figure;
stem(rho)
ylabel('\rho(h)')
xlabel('h')
title('Autocorrelation')

figure;
plot(gamma)
ylabel('\gamma(h)')
xlabel('h')
title('Autocovariance')

sz = size(gamma);
a = ones(sz);
a(a == 1) = 1;



%% FILL MISSING DATA AND COMPUTE MSE ON Y

close all
clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;
complete_data = table.Volume;

q = 4;

Y_missing = computeLogReturns(data);
NaN_idx = find(isnan(Y_missing));

rho = autocorr(data, length(Y_missing)-length(NaN_idx));
rho_mat = zeros(q-1, q-1);

% Init rho-matrix
for i = 1:q-1
    for j = 1:q-1
        rho_mat(i, j) = rho(1 + abs(i - j));
    end
end

a_vec = linsolve(rho_mat, rho(2:q));

data_notNaN = data(~isnan(data));
mu = mean(data_notNaN);
a0 = mu * (1 - sum(a_vec));

n_NaN = 0;

for i = 1:length(data)
    if isnan(data(i))
        n_NaN = n_NaN + 1;
        data(i) = computePred(data, i, a_vec, a0, q);
    end
end

missing = computeLogReturns(data);
complete = computeLogReturns(complete_data);
total_error = sum((complete(NaN_idx) - missing(NaN_idx)).^2);

figure;
plot(complete(NaN_idx))
%title('Complete Data')
hold on

plot(missing(NaN_idx))
title('Data with Missing Values Predicted')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')


% Percentage error
error = sqrt(total_error / length(NaN_idx))

%% Problem 3 - Predict on Y

close all
clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;
complete_data = table.Volume;

q = 7;

Y_missing = computeLogReturns(data);
Y_complete = computeLogReturns(complete_data);

rho = autocorr(Y_missing);
rho_mat = zeros(q-1, q-1);

% Init rho-matrix
for i = 1:q-1
    for j = 1:q-1
        rho_mat(i, j) = rho(1 + abs(i - j));
    end
end

a_vec = linsolve(rho_mat, rho(2:q));

data_notNaN = Y_missing(~isnan(Y_missing));
NaN_idx = find(isnan(Y_missing));

mu = mean(data_notNaN);

a0 = mu * (1 - sum(a_vec));

n_NaN = 0;

for i = 1:length(Y_missing)
    if isnan(Y_missing(i))
        n_NaN = n_NaN + 1;
        Y_missing(i) = computePred(Y_missing, i, a_vec, a0, q);
    end
end

total_error = sum((Y_complete(NaN_idx) - Y_missing(NaN_idx)).^2);

figure;
plot(Y_complete(NaN_idx))
%title('Complete Data')
hold on

plot(Y_missing(NaN_idx))
title('Data with Missing Values Predicted')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')


% Percentage error
error = sqrt(total_error / n_NaN)

%% Problem 3 - ULTIMATE EDITION GOLD PLUS - 1699 kr - Only On Playstation

close all
clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;
complete_data = table.Volume;

q = 4;

Y_missing = computeLogReturns(data);
Y_complete = computeLogReturns(complete_data);

NaN_idx = find(isnan(Y_missing));

rho = autocorr(data, length(Y_missing)-length(NaN_idx));
N = length(rho);

for idx = 1:length(NaN_idx)

    s = computeS(NaN_idx(idx), NaN_idx, q, N);

    rho_mat = zeros(length(s), length(s));
    for i = 1:length(s)
        for j = 1:length(s)
            rho_mat(i, j) = rho(abs(s(end+1-j) - s(end+1-i))+1);
        end
    end

    a_vec = linsolve(rho_mat, rho(1+abs(NaN_idx(idx)-flip(s))));

    data_notNaN = Y_missing(~isnan(Y_missing));
    mu = mean(data_notNaN);

    a0 = mu * (1 - sum(a_vec));
    
    Y_missing(NaN_idx(idx)) = computePred(Y_missing, s, a_vec, a0);
end

total_error = sum((Y_complete(NaN_idx) - Y_missing(NaN_idx)).^2);

figure;
plot(Y_complete(NaN_idx))
%title('Complete Data')
hold on

plot(Y_missing(NaN_idx))
title('Data with Missing Values Predicted')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')


% Percentage error
error = sqrt(total_error / 200)

%% Functions

function s = computeS(idx, NaN_idx, q, N)
    t = idx;
    s = max([1, t - q]) : min([N - 1, t + q]);
    s = setdiff(s, NaN_idx);
    
end

function rho_mat = computeRhoMat(idx, q)
rho_mat = zeros(2*q-1, 2*q-1);
for i = idx-q/2:2*q-1
    for j = 1:2*q-1
        rho_mat(i, j) = rho(1 + abs(i - j));
    end
end
end

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

function pred = computePred(Y_missing, s, a_vec, a0)

datapoints = Y_missing(s);

pred = a0 + dot(a_vec, fliplr(datapoints));

end

function[gamma] = autocov(X,Y)

X = X(~isnan(Y));
Y = Y(~isnan(Y));

% Verify the input consistency
if nargin < 1, error('Missing input vector.'); end

[M N] = size(X);
[P Q]= size(Y);
if M < 2, error('X is too short.'); end
if M~=P || N~=Q, error('Input vectors do not have the same size.'); end
if N ~= 1, error('X must be a column vector.'); end

% Compute the autocovariance
X_pad = [X - mean(X); zeros(M,1)];%%paddle
Y_pad = [Y - mean(Y); zeros(M,1)];

X_hat = fft(X_pad);
Y_hat = fft(Y_pad);

gamma = ifft( conj( X_hat ) .* Y_hat );%the imaginary part is due to float precision errors.
gamma = real( gamma(1:M-1) ) ./ (M - (1:1:M-1))';

end