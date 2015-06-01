addpath_scatnet;

x = uiuc_sample;

x = x(1:129, 1:129);

J = 3;
Q = 1;
L = 4;

filter_opt.J = J;
filter_opt.Q = Q;
filter_opt.L = L;
filter_opt.filter_type = 'morlet';

W_op = wavelet_factory_2d(size(x), filter_opt);

S = scat(x, W_op);

clear W_op

save scattering_transformed_image.mat -v7
