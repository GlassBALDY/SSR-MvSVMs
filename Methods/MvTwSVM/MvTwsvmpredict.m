function [bina_result, origin_viewAp, origin_viewAn, origin_resultA, origin_viewBp, origin_viewBn, origin_resultB, model] = MvTwsvmpredict(model, testXA, testXB, testy)
%	THis function is used to implement MvTwinsvm prediction;
% Input:
%	model: a multi-view twinsvm model trained by Twinsvmtrain;
%	testXA/B: viewA/B's test set or the sample you want to predict;
%   testy: test set's real label;
% Output:
%	bina_result: predicted result = +1, -1;
%   origin_viewA/Bp: viewA/B's predicted result = w+*x + b+;
%   origin_viewA/Bn: viewA/B's predicted result = w-*x + b-.
% Usage:
%   [~, ~, ~, ~, ~, ~, ~, model] = MvTwsvmpredict(model, testXA, testXB, testy)

% - Jiayi Zhu, 23rd September 2020
% - Jiayi Zhu, 30th March 2021
    tic;
    %% Import parameters from model
    % obtain the number of test pattern
    mta = size(testXA, 1);    
    mtb = size(testXB, 1); 
    
    % obtain the viewA/B's train set
    CA = model.CA;            
    CB = model.CB;
    
    % obtain the viewA/B's kerneltype and kernelparameters
    kerneltypeA = model.kerneltypeA;
    kernelparamA = model.kernelparamA;
    kerneltypeB = model.kerneltypeB;
    kernelparamB = model.kernelparamB;
    
    % import viewA/B's positive hyperplane and negative hyperplane
    uAp = model.uAp; uBp = model.uBp;
    vAn = model.vAn; vBn = model.vBn;
    
    %% Calculate viewA/B's binary result
    % calculate viewA/B's kernel matrix
    kermatA = [kernelfunction(kerneltypeA, testXA, CA, kernelparamA) ones(mta, 1)];
    kermatB = [kernelfunction(kerneltypeB, testXB, CB, kernelparamB) ones(mtb, 1)];
    
    % viewA's origin result
    origin_viewAp = kermatA * uAp;                              % w+ * x + b+
    origin_viewAn = kermatA * vAn;                              % w- * x + b-
    
    % viewB's origin result
    origin_viewBp = kermatB * uBp;                              % w+ * x + b+
    origin_viewBn = kermatB * vBn;                              % w- * x + b-
    
    if length(origin_viewAp) == length(origin_viewBp)   % if the sample have two view, then do two view decision
        origin_resultA = abs(origin_viewAn)/norm(vAn(1:(end-1))) - abs(origin_viewAp)/norm(uAp(1:(end-1)));
        if isnan(origin_resultA) == 1
            origin_resultA = zeros(size(origin_viewAn));
        end
        
        origin_resultB = abs(origin_viewBn)/norm(vBn(1:(end-1))) - abs(origin_viewBp)/norm(uBp(1:(end-1)));
        if isnan(origin_resultB) == 1
            origin_resultB = zeros(size(origin_viewBn));
        end
        
%         bina_positive = abs(origin_viewAp)/norm(uAp(1:(end-1))) + abs(origin_viewBp)/norm(uBp(1:(end-1)));   % |w+ * x + b+|/||w+||
%         bina_negative = abs(origin_viewAn)/norm(vAn(1:(end-1))) + abs(origin_viewBn)/norm(vBn(1:(end-1)));   % |w- * x + b-|/||w-||
        origin_result = origin_resultA + origin_resultB;              % |w- * x + b-|/||w-|| - |w+ * x + b+|/||w+||
        bina_result = sign(origin_result);
        bina_A = sign(origin_resultA);
        bina_B = sign(origin_resultB);
    else
        origin_resultA = abs(origin_viewAn)/norm(vAn(1:(end-1))) - abs(origin_viewAp)/norm(uAp(1:(end-1)));
        origin_resultB = abs(origin_viewBn)/norm(vBn(1:(end-1))) - abs(origin_viewBp)/norm(uBp(1:(end-1)));
        bina_result = [];                   % else, do not
        bina_A = sign(origin_resultA);
        bina_B = sign(origin_resultB);
    end
    toc;
    time = toc;
    if nargin == 4              % if testy true
        model.testXA = testXA;
        model.testXB = testXB;
        model.testy = testy;
        model.testing_time = time;
        [model.accuracy, model.testresult] = judgement(bina_result, testy);
        [model.accuracy_viewA, model.testresult_viewA] = judgement(bina_A, testy);
        [model.accuracy_viewB, model.testresult_viewB] = judgement(bina_B, testy);
    end
end