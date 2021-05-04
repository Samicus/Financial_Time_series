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



%% PREDICT FUTURE VALUES AND COMPUTE MSE ON Y (not part of task)

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
hold on
plot(missing(NaN_idx))
title('Data with Missing Values Predicted')
xlabel('NaN Indices')
ylabel('Value')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')

% Percentage error
error = sqrt(total_error / length(NaN_idx))

%% Problem 3 - Predict on X

close all
clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;
complete_data = table.Volume;

Y_missing2 = computeLogReturns(data);
NaN_idx2 = find(isnan(Y_missing2));

q = 4;

NaN_idx = find(isnan(data));

rho = autocorr(data, length(data) - length(NaN_idx) - 1);
N = length(data);

data_notNaN = data(~isnan(data));
mu = mean(data_notNaN);

for idx = 1:length(NaN_idx)

    s = computeS(NaN_idx(idx), NaN_idx, q, N);

    rho_mat = computeRhoMat(s, rho);

    a_vec = linsolve(rho_mat, rho(1+abs(NaN_idx(idx)-fliplr(s))));

    a0 = mu * (1 - sum(a_vec));
    
    data(NaN_idx(idx)) = computePred(data, flip(s), a_vec, a0);
end

Y_missing = computeLogReturns(data);
Y_complete = computeLogReturns(complete_data);

total_error = sum((Y_complete(NaN_idx2) - Y_missing(NaN_idx2)).^2);

figure;
plot(Y_complete(NaN_idx2))
hold on
plot(Y_missing(NaN_idx2))
title('Data with Missing Values Predicted on X')
xlabel('NaN Indices')
ylabel('Value')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')


% Percentage error
error = sqrt(total_error / length(NaN_idx2))

%% Problem 3 - Predict on Y

close all
clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;
complete_data = table.Volume;

q = 4;

Y_missing = computeLogReturns(data);
Y_interpolation = computeLogReturns(data);
Y_complete = computeLogReturns(complete_data);

NaN_idx = find(isnan(Y_missing));

rho = autocorr(Y_missing, length(Y_missing) - length(NaN_idx) - 1);
N = length(data);

data_notNaN = Y_missing(~isnan(Y_missing));
mu = mean(data_notNaN);

for idx = 1:length(NaN_idx)

    s = computeS(NaN_idx(idx), NaN_idx, q, N);

    rho_mat = computeRhoMat(s, rho);

    a_vec = linsolve(rho_mat, rho(1+abs(NaN_idx(idx)-fliplr(s))));

    a0 = mu * (1 - sum(a_vec));
    
    Y_missing(NaN_idx(idx)) = computePred(Y_missing, flip(s), a_vec, a0);
end

total_error = sum((Y_complete(NaN_idx) - Y_missing(NaN_idx)).^2);

figure;
plot(Y_complete(NaN_idx))
hold on
plot(Y_missing(NaN_idx))
title('Data with Missing Values Predicted on Y')
xlabel('NaN Indices')
ylabel('Value')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')


% Percentage error
error = sqrt(total_error / length(NaN_idx))

[F,TF] = fillmissing(Y_interpolation,'linear','SamplePoints',1:length(Y_interpolation));
total_error_interp = sum((Y_complete(NaN_idx) - F(TF)).^2);
error = sqrt(total_error_interp / length(NaN_idx))

figure;
plot(Y_complete(NaN_idx))
hold on
plot(F(TF))
title('Data with Interpolation')
xlabel('NaN Indices')
ylabel('Value')
legend({'Actual Values', 'Predicted Values'},'Location','southwest')

%% Functions

function s = computeS(idx, NaN_idx, q, N)
    t = idx;
    s = max([1, t - q]) : min([N - 1, t + q]);
    s = setdiff(s, NaN_idx);
    
end

function rho_mat = computeRhoMat(s, rho)
rho_mat = zeros(length(s), length(s));
    for i = 1:length(s)
        for j = 1:length(s)
            rho_mat(i, j) = rho(abs(s(end+1-j) - s(end+1-i))+1);
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