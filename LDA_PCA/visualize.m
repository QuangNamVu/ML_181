clc;
close all;

N = 4;

N1 = N2 = N/2;

# x - y = c
# => normal vector = [1 -1]
W = [1, -1];

grad = -W(2)/W(1);
b = 3;

X1 = 3*rand(1, N1);
X2 = 3*rand(1, N2);
Y1 = grad*X1 + randn(1, N1) + b;
Y2 = grad*X2 + randn(1, N2) - b;
figure
plot(X1, Y1, 'b+');hold on;
plot(X2, Y2, 'ro');hold on;

mu1 = [mean(X1), mean(Y1)];

mu2 = [mean(X2), mean(Y2)];

mu = [mean([X1, X2]), mean([Y1, Y2])];

# plot mu
plot(mu1(1), mu1(2)  , 'k+', 'markerSize', 20, 'linewidth', 3); 
hold on
plot(mu2(1), mu2(2), 'ko', 'markerSize', 20, 'linewidth', 3); 
hold on
plot(mu(1), mu(2), 'k*', 'markerSize', 20, 'linewidth', 3); 
hold on


S1 = cov(X1);
S2 = cov(X2);
SW = S1+S2;

W = inv(SW)*(mu2-mu1)

scale1 = 1.5;
scale2 = .75;


line1 = line([mu(1)-scale2*W(1) mu(1)+scale2*W(1)],
             [mu(2)-scale2*W(2) mu(2)+scale2*W(2)]); 
             
line2 = line([mu(1)-scale2*W(2) mu(1)+scale2*W(2)],
             [mu(2)-scale2*W(1) mu(2)+scale2*W(1)]);             
        
set(line1, 'color', [1 0 0], "linestyle", "--")
set(line2, 'color', [0 1 0], "linestyle", "--")
# plot point to line

X1_shift = X1 - mu(1);
X2_shift = X2 - mu(1);


Y1_shift = Y1 - mu(2);
Y2_shift = Y2 - mu(2);

best_line_idx = 2;
X1_new = X1_shift*W(best_line_idx);
X2_new = X2_shift*W(best_line_idx);

Y1_new = X1_new*W(best_line_idx)' + mu(2);
Y2_new = X2_new*W(best_line_idx)' + mu(2);


# point_red = plot(X1,Y1_new,"ro", "markersize", 10, "linewidth", 3);