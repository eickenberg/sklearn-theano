addpath_scatnet

N = 48;
M = 64;

sigmas = [1, 2, 4, 8];
slants = [1., .5, .25];
xis = [.25, .5, 1., 2., 4.];
thetas = [0., pi / 8., pi / 4., pi / 2., 5. * pi / 8., 7. * pi / 4.];

filters = zeros(N, M,
		length(sigmas), length(slants),
		length(xis), length(thetas));


for i=1:length(sigmas)
	sigma = sigmas(i)
	for j=1:length(slants)
		slant=slants(j)
		for k=1:length(xis)
			xi=xis(k)
			for l=1:length(thetas)
				%i
				%j
				%k
				%l
				theta=thetas(l)
				fflush(stdout)
				fil = morlet_2d_noDC(N, M, sigma, slant, xi, theta);

                                filters(:, :, i, j, k, l) = fil;
			 end
		end
	end
end

save morlet_filters_noDC.mat -v7

