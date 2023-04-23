function [] = myscatter( vector, dot_type)
%   Automatically convert a 2-D matrix into a dot plot
% Input:
%   vector: a matrix with dimension n-by-2;
%   dot_type: a char which assign the style of the dot.
% Output:
%   the graphic we need

% ===== main function ================================
    if nargin == 1
        % if the dot_style is not assigned, do a default style.
        plot(vector(:, 1), vector(:, 2), '.');
    else
        % other wise, do the assigned style.
        plot(vector(:, 1), vector(:, 2), dot_type);
    end
end

