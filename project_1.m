data = readtable('intel.csv');
data_v = data.Volume
n = height(data.Volume);
mx = mean(data.Volume);
Y = zeros(1,n-1)
for i=1:n-1
    Y(i) = log(data_v(i+1)) - log(data_v(i));
end
lags = 30;
gamma = zeros(1, lags+1);
for h=0:lags
   gamma(h+1) = dot((Y(1+h:end)-mx),(Y(1:end-h)-mx)') / n;
end

acf = gamma/gamma(1)


stem(acf)
axis([-1 lags+2 -1.5 1.5])