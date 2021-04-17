%% Complete Data

clear
clc

table = readtable('intel.csv');
data = table.Volume;

q = 7;

X_tp1    = data(2 : end);
X        = data(1 : end-1);

returns         = X_tp1 - X;
log_returns     = log(X_tp1) - log(X);

[rho, lags, bounds, h] = autocorr(log_returns);

gamma = autocov(X_tp1, log_returns);
gamma = gamma(1:q);

%% Incomplete Data

clear all
clc

table = readtable('intel.csv');
data = table.VolumeMissing;

q = 7;

X_tp1 = data(2 : end);
X = data(1 : end-1);

returns = X_tp1 - X;
log_returns = log(X_tp1) - log(X);

[rho, lags, bounds, h] = autocorr(log_returns);

gamma = autocov(X_tp1, log_returns);
gamma = gamma(1:q);

%% Problem 2
close all

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


%% Problem 3

clc
close all


q = 7;

rho_mat = zeros(q, q);

for i = 1:q
    for j = 1:q
        rho_mat(i, j) = rho(q + 1 - j) - rho(q + 1 - i);
    end
end

a_mat = rho_mat * rho(1:q)
plot(a_mat)

%% Functions

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