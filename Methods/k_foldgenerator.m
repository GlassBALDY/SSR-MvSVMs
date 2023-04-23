function [indices] = k_foldgenerator(k, m)
%   This function is used to generate k-fold validation sequence
% Input:
%   k: number of fold;
%   m: number of samples;
%   i; the ith fold that will be the test set;
% Output:
%q	indices: a indices vector which will be used for k-fold validatio ;

% ========= main function =======================================
	t = floor(m/k);	% obtain the repeat time
	b = mod(m, k);	% obtain the remainder
	seq = 1:k;
	indices_rep = repmat([1:k]', t, 1);
	indices_remain = [1:b]';
	indices = [indices_rep; indices_remain];
end

