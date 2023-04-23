function [ trainX, trainy, testX, testy ] = train_test_split( X, y, test_size, seed )
%   separate a random train set and test set
% Input:
%   X: Instances
%   y: labels
%   test_size: size of test instances
%   seed: random seed for random split
% Output:
%   trainX: train data
% trainy: train label
% testX: test data
% testy: test label

% ============= main function ===============
    if nargin == 4
        rng(seed);
    end
    idx = randperm(length(y));
    split_point = floor(length(y) * (1 - test_size));
    trainX = X(idx(1:split_point), :);
    trainy = y(idx(1:split_point), :);
    testX = X(idx(split_point+1:length(y)), :);
    testy = y(idx(split_point+1:length(y)), :);
end

