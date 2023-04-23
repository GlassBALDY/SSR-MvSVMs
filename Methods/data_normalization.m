function [fixedX] = data_normalization(X, method)
%   This function is used to implement 0-1 normalization
% Input:
%   X: data you want to be normalized with n-by-m vector, n is the number
%   of samples and m is the dimension.
% Output:
%   fixedX: normalized data
%   Example:
%   [fixedX] = data_normalization(X, '01')

% ============= main function =============================================
    [num, dim] = size(X);
    switch method
        case '01'
        for i = 1:dim
            minimum = min(X(:, i)); maximum = max(X(:, i));
            for j = 1:num
                if minimum ~= maximum
                    fixedX(j, i) = (X(j, i) - minimum)/(maximum - minimum);
                else
                    fixedX(j, i) = 0;
                end
            end
            
        end
        
        case '-11'
            fixedX = [mapminmax(X')]';
        
        case 'Zscore'
            fixedX = zscore(X);
        
        case 'default'
            fixedX = X;
    end
end

