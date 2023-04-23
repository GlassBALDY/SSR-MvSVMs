function [bina_result, origin_viewA, origin_viewB, model] = svm2kpredict(model, test_set1, test_set2, testy)
%   This function is used to implement the svm2k classification.
% Input:
%   model: the svm2k classifier model trained by svm2ktrain.m 
%   test_set1: viewA's test set with n-by-p, p is the number of attributes;
%   test_set2: viewB's test set with n-by-q, q is the number of attributes;
% Output:
%   bina_result: predicted result = +1, -1;
%   origin_result: predicted result = w*x + b.
% example:
%   [bina_result, origin_viewA, origin_viewB, model] = svm2kpredict(model, test_set1, test_set2, testy)
% Notation:
%   在这个SVM-2K训练中，我们使用了“样本增广”（暂时如此称呼），即给原来的样本尾部加1，x -> [x, 1]
%   令X = [x, 1], W = [w, b];
%   如此一来，w'x + b = <w, x> + b <==> [w, b]'[x, 1]，
%   而ker(X, X) = ker(x, x) + 1;

% - Jiayi Zhu 19th August 2020
% - update by Jiayi Zhu, 26th September 2020
% - update by Jiayi Zhu, 15th September 2021

% ====== main function ====================================================
    tic
    if nargin < 4
        ga = model.ga;
        gb = model.gb;
        typeA = model.kerneltypeA;
        typeB = model.kerneltypeB;
        CA = model.CA;
        CB = model.CB;
        sigmaA = model.kernelparamA;
        sigmaB = model.kernelparamB;
        % calculate b
        sv_a = model.support_vec_a;
        sv_b = model.support_vec_b;
        sv_y1 = model.support_y1;
        sv_y2 = model.support_y2;
        
        % obtain the number of test set
        nteA = size(test_set1, 1); nteB = size(test_set2, 1);
        % obtain the number of train set
        ntrA = size(CA, 1); ntrB = size(CB, 1);
        
        %{
        % b = y - w'x; w = y.*g*x; w'x = y.*g*k(x,x)
        ba = mean(sv_y1 - (ga' * kernelfunction(typeA, CA, sv_a, sigmaA))');
        bb = mean(sv_y2 - (gb' * kernelfunction(typeB, CB, sv_b, sigmaB))');
        %}
        
        if isempty(test_set2) == 1          % if there is only view A's data
            origin_viewA = [kernelfunction(typeA, test_set1, CA, sigmaA) + 1] * ga;
            origin_viewB = [];
            bina_result = sign(origin_viewA);
        elseif isempty(test_set1) == 1      % if there is only view B's data
            origin_viewB = [kernelfunction(typeB, test_set2, CB, sigmaB) + 1] * gb;  
            origin_viewA = [];
            bina_result = sign(origin_viewB);
        else
            origin_viewA = [kernelfunction(typeA, test_set1, CA, sigmaA) + 1] * ga;
            origin_viewB = [kernelfunction(typeB, test_set2, CB, sigmaB) + 1] * gb;
            bina_result = sign(origin_viewA + origin_viewB);
        end
        %{
        model.ba = ba;
        model.bb = bb;
        %}
    else 
        ga = model.ga;
        gb = model.gb;
        typeA = model.kerneltypeA;
        typeB = model.kerneltypeB;
        CA = model.CA;
        CB = model.CB;
        sigmaA = model.kernelparamA;
        sigmaB = model.kernelparamB;
        % calculate b
        sv_a = model.support_vec_a;
        sv_b = model.support_vec_b;
        sv_y1 = model.support_y1;
        sv_y2 = model.support_y2;
        
        % obtain the number of test set
        nteA = size(test_set1, 1); nteB = size(test_set2, 1);
        % obtain the number of train set
        ntrA = size(CA, 1); ntrB = size(CB, 1);
        
        % b = y - w'x; w = y.*g*x; w'x = y.*g*k(x,x)
        ba = mean(sv_y1 - (ga' * kernelfunction(typeA, CA, sv_a, sigmaA))');
        bb = mean(sv_y2 - (gb' * kernelfunction(typeB, CB, sv_b, sigmaB))');

        if isempty(test_set2) == 1          % if there is only view A's data
            origin_viewA = [kernelfunction(typeA, test_set1, CA, sigmaA) + 1] * ga;
            origin_viewB = [];
            bina_result = sign(origin_viewA);
        elseif isempty(test_set1) == 1      % if there is only view B's data
            origin_viewB = [kernelfunction(typeB, test_set2, CB, sigmaB) + 1] * gb;  
            origin_viewA = [];
            bina_result = sign(origin_viewB);
        else
            origin_viewA = [kernelfunction(typeA, test_set1, CA, sigmaA) + 1] * ga;
            origin_viewB = [kernelfunction(typeB, test_set2, CB, sigmaB) + 1] * gb;
            bina_A = sign(origin_viewA); bina_B = sign(origin_viewB);
            bina_result = sign(origin_viewA + origin_viewB);
        end
        %{
        model.ba = ba;
        model.bb = bb;
        %}
        time = toc;
        model.testXA = test_set1;
        model.testXB = test_set2;
        model.testy = testy;
        model.testing_time = time;
        [model.accuracy_viewA, model.testresult_viewA] = judgement(bina_A, testy);
        [model.accuracy_viewB, model.testresult_viewB] = judgement(bina_B, testy);
        [model.accuracy, model.testresult] = judgement(bina_result, testy);
    end
end