addpath_scatnet;

options.filter_format = 'fourier';

Q = 1;
J = 4;
L = 8;

options.Q = Q;
options.J = J;
options.L = L;

N = 48;
M = 64;

filters = morlet_filter_bank_2d([N, M], options);

save morlet_pyramid_noDC.mat -v7

