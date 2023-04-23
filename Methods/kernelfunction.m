function[k] = kernelfunction(type, u, v, p)

% Function to compute kernel
% This function computes linear, polynomial and RBF kernels
% Inputs:
% type: Kernel type (1:linear, 2:polynomial, 3:RBF
% u: Sample x_i for computing kernel
% v: Sample x_j for computing kernel
% p: Kernel parameter (degree for polynomial, kernel width for RBF

% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in
% reformed by Jiayi Zhu, January 24th 2021.

switch type
    
    case 'linear'
        %Linear Kernal
        k = u*v';
        
    case 'poly'
        %Polynomial Kernal
        k = (u*v' + 1)^p;
        
    case 'rbf'
        %Radial Basia Function Kernal
        m = size(u, 1);
        n = size(v, 1);
        G = zeros(m, n);
        for i = 1:m
            for j = 1:n
                temp = u(i, :) - v(j, :);
                t = temp * temp';
                G(i, j) = exp(-t/(2 * p^2));
            end
        end
        k = G;

    case 'Gauss'
        p = p*p;
        uu = sum(u.*u,2);
        vv = sum(v.*v,2);
        uv = u*v';
        K = [repmat(uu,[1 size(vv,1)]) + repmat(vv',[size(uu,1) 1]) - 2*uv];
        k = exp(-K./(2*p));

    case 'Gauss_image'
        % this kernel add a scalar before the original Gauss kernel
        scaler = 1 / (sqrt(2 * pi) * p);
        p = p*p;
        uu = sum(u.*u,2);
        vv = sum(v.*v,2);
        uv = u*v';
%         K = [repmat(uu,[1 size(vv,1)]) + repmat(vv',[size(uu,1) 1]) - 2*uv];
        K = [repmat(uu,[1 size(vv,1)]) + repmat(vv',[size(uu,1) 1])];
        k = scaler .* exp(-K./(2*p));
    otherwise
        k=0;      
end
return
