function [k] = diagkernelfunction(type, u, p)
%   Function to compute kernel matrix's diagonal
%   This function computes linear, polynomial and RBF kernels
% Inputs:
%   type: Kernel type (1:linear, 2:poly, 3:rbf
%   u: Sample x_i for computing kernel
%   v: Sample x_j for computing kernel
%   p: Kernel parameter (degree for polynomial, kernel width for RBF
% Output:
%   k: kernel matrix

% Sumit Soman, 20 July 2018
% eez127509@ee.iitd.ac.in

% Reformed by Jiayi, 18 August 2020
% - Update by Jiayi Zhu, 11th October 2020
% - Update by Jiayi Zhu, 27th January 2021
    switch type

        case 'linear'
            %Linear Kernal
            k = sum(u .* u, 2);

        case 'poly'
            %Polynomial Kernal  
            k = (sum(u .* u, 2) + 1).^p;

        case 'rbf'
            %Radial Basia Function Kernal
            k = ones(size(u, 1), 1);      
        case 'Gauss'
            %Radial Basia Function Kernal
            k = ones(size(u, 1), 1); 
    end
return


