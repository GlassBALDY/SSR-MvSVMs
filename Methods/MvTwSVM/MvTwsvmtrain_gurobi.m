function [ model ] = MvTwsvmtrain_gurobi(XA, XB, y, Box_Cons_p, Box_Cons_n, D, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon, varargin)
%   This function is used to implement Multi-view Twin SVM.
% Input:
%   XA: viewA's train data;
%   XB: viewB's train data;
%   y: train data's label;
%   Box_Cons_p/n: viewA/B's penalty parameter vector with 2-by-1,
%   Box_Cons_p/n(1) = viewA's C_p/n, Box_Cons_p/n(2) = viewB's C_p/n
%   D: penalty parameter vector for similarity constraint's slacks with 2-by-1.
%   kerneltypeA/B: viewA/B's kerneltype, kerneltypeA/B = 'linear', 'rbf', 'poly'
%   kernelparamA/B: viewA/B's kernel parameter;
%   epsilon: similarity constraint
% Alternative parameters:
%   'Tikhonov': Tikhonov regulization parameter, default = 1e-8
%
% Output:
%   model:  Multi-view Twin SVM model we trained.
% Usage:
%   [ model ] = MvTwsvmtrain(XA, XB, y, Box_Cons_p, Box_Cons_n, D, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon, 'Tikhonov', 1e-8)

% - Jiayi Zhu 20th September 2020  
% - reformed by Jiayi Zhu 23rd September 2020
% - update by Jiayi Zhu, 24th October 2020
% - adjusting the parameter ebsilon = 1e-8 to ebsilon = 1e-5;
% - adding varargin.

%   Reference:
% Xie, Xijiong , and S. Sun . 
% "Multi-view twin support vector machines." 
% Intelligent Data Analysis 19.4(2015):701-712.

    num_accuracy = 1e-7; 
    num_truncation = 6;
    global global_options
% ======== main function ================================================
    % parameter initialization
    Tikhonov = 1e-5;
    
    % parameter detection
    p = inputParser;
    addParameter(p, 'Tikhonov', Tikhonov);
    parse(p, varargin{:});
    
    % read parameter
    ebsilon = p.Results.Tikhonov;   % a very tiny Tikhonov regulization term
    % obtain the +1 pattern and -1 pattern for each view
    XAp = XA(y == 1, :);
    XAn = XA(y == -1, :);
    XBp = XB(y == 1, :);
    XBn = XB(y == -1, :);
    CA = [XAp; XAn];
    CB = [XBp; XBn];
    
    tic
    % obtain the number of +1 patterns and -1 patterns
    l_p = size(XAp, 1); l_n = size(XAn, 1);
    E_n = eye(l_n); O_n = zeros(l_n); E_p = eye(l_p); O_p = zeros(l_p); O_pn = zeros(l_p, l_n); O_np = zeros(l_n, l_p);
    
    % create the quadratic matrix
    E1 = [kernelfunction(kerneltypeA, XAp, CA, kernelparamA), ones(l_p, 1)];
    F1 = [kernelfunction(kerneltypeB, XAn, CA, kernelparamB), ones(l_n, 1)];
    E2 = [kernelfunction(kerneltypeA, XBp, CB, kernelparamA), ones(l_p, 1)];
    F2 = [kernelfunction(kerneltypeB, XBn, CB, kernelparamB), ones(l_n, 1)];
    cesi1 = [F1' * [-E_n O_n], E1' * [-E_p E_p]];
    cesi2 = [F2' * [O_n -E_n], E2' * [E_p -E_p]];
    rol1 = [E1' * [E_p O_p], F1' * [-E_n E_n]];
    rol2 = [E2' * [O_p E_p], F2' * [E_n -E_n]];
    
    % obtain viewA/B's postive/negative inv matrix
    E1TE1 = E1' * E1; E2TE2 = E2' * E2; F1TF1 = F1' * F1; F2TF2 = F2' * F2;
    invE1TE1 = inv(E1TE1 + ebsilon * eye(size(E1TE1, 1)));
    invE2TE2 = inv(E2TE2 + ebsilon * eye(size(E2TE2, 1)));
    invF1TF1 = inv(F1TF1 + ebsilon * eye(size(F1TF1, 1)));
    invF2TF2 = inv(F2TF2 + ebsilon * eye(size(F2TF2, 1)));
    
    Z_p = [chol(E1TE1 + ebsilon * eye(size(E1TE1, 1))), zeros(size(E1TE1));
           zeros(size(E2TE2)), chol(E2TE2 + ebsilon * eye(size(E2TE2, 1)))];
    Z_n = [chol(F1TF1 + ebsilon * eye(size(F1TF1, 1))), zeros(size(F1TF1));
           zeros(size(F1TF1)), chol(F2TF2 + ebsilon * eye(size(F2TF2, 1)))];

    %%
    % solve the first QPP - E
    quad_obj_p = cesi1' * invE1TE1 * cesi1 + cesi2' * invE2TE2 * cesi2;
    linear_obj_p = [- ones(1, 2 * l_n), epsilon * ones(1, 2 * l_p)]';     % obtain linear vector
    
    % ineq_cons
    A = [O_pn O_pn E_p E_p];   
    B = D(1) * ones(l_p, 1);  

    % eq_cons
    Aeq = [];  
    Beq = [];
    
    % boundary
    if isempty(Box_Cons_p) == 0   % if Box_Cons true, reset the upper bound of lagrange multiplier
        LB = zeros(2 * (l_n + l_p), 1); 
        UB = [ones(l_n, 1) * Box_Cons_p(1); ones(l_n, 1) * Box_Cons_p(2); D(1) * ones(2 * l_p, 1)];  
    else
        LB = zeros(2 * (l_n + l_p), 1);  
        UB = [];                % otherwise, there is no limit to lagrange multiplier
    end
    
    %% gurobi's data structure
    % objective 
    model_p.Q = 0.5 * sparse(quad_obj_p);
    model_p.obj = linear_obj_p;    
    
    % ineq_cons
    model_p.A = sparse(A);   
    model_p.rhs = B;
    model_p.sense = '<';

    % box bound
    model_p.lb = LB;
    model_p.ub = UB;    
    
    % variable type:(c)ontinuous
    model_p.vtype = 'C';
    % goal: minimization
    model_p.modelsense = 'min';
    
    % solve it
    try
        result_p = gurobi(model_p);
        pai = result_p.x;
        pai = round(pai, num_truncation);
    catch
        [ pai ] = quadprog(quad_obj_p, linear_obj_p, A, B, Aeq, Beq, LB, UB, [], global_options);
        pai = round(pai, num_truncation);
    end
    % this is the key to generate decision hyper-plane
    % so we do a very strong numeric truncation
    lag_paiA = cesi1 * pai;
    lag_paiB = cesi2 * pai;
    
    % calculate positive decision parameters
    uAp = invE1TE1 * lag_paiA;
    uBp = invE2TE2 * lag_paiB;
    
    model.pos.obj_quad = quad_obj_p; 
    model.pos.obj_linear = linear_obj_p;
    model.pos.A = A;
    model.pos.B = B;
    model.pos.UB = UB;
    model.pos.pi = pai;
    model.pos.cesi_A = cesi1;
    model.pos.cesi_B = cesi2;
    model.pos.inv_A = invE1TE1;
    model.pos.inv_B = invE2TE2;
%     model.pos.Z = Z_p;
    
    %%
    % solve the second QPP - F
    quad_obj_n = rol1' * invF1TF1 * rol1 + rol2' * invF2TF2 * rol2;
    linear_obj_n = [- ones(1, 2 * l_p) epsilon * ones(1, 2 * l_n)]';     % obtain linear vector
    
    % ineq_cons
    A = [O_np O_np E_n E_n];   
    B = D(2) * ones(l_n, 1);  

    % eq_cons
    Aeq = [];  
    Beq = [];
    
    % boundary
    if isempty(Box_Cons_n) == 0   % if Box_Cons true, reset the upper bound of lagrange multiplier
        LB = zeros(2 * (l_n + l_p), 1); 
        UB = [ones(l_p, 1) * Box_Cons_n(1); ones(l_p, 1) * Box_Cons_n(2); D(2) * ones(2 * l_n, 1)];
    else
        LB = zeros(2 * (l_n + l_p), 1);  
        UB = [];                % otherwise, there is no limit to lagrange multiplier
    end
    
    %% gurobi's data structure
    % objective 
    model_n.Q = 0.5 * sparse(quad_obj_n);
    model_n.obj = linear_obj_n;    
    
    % ineq_cons
    model_n.A = sparse(A);   
    model_n.rhs = B;
    model_n.sense = '<';

    % box bound
    model_n.lb = LB;
    model_n.ub = UB;    
    
    % variable type:(c)ontinuous
    model_n.vtype = 'C';
    % goal: minimization
    model_n.modelsense = 'min';
    
    % solve it
    try
        result_n = gurobi(model_n);
        fai = result_n.x;
        fai = round(fai, num_truncation);
    catch
        [ fai ] = quadprog(quad_obj_n, linear_obj_n, A, B, Aeq, Beq, LB, UB, [], global_options);
	    fai = round(fai, num_truncation);
    end
    % this is the key to generate decision hyper-plane
    % so we do a very strong numeric truncation
    lag_faiA = rol1 * fai;
    lag_faiB = rol2 * fai;

    % calculate negative decision parameters
    vAn = invF1TF1 * lag_faiA;
    vBn = invF2TF2 * lag_faiB;
    
    model.neg.obj_quad = quad_obj_n; 
    model.neg.obj_linear = linear_obj_n;
    model.neg.A = A;
    model.neg.B = B;
    model.neg.UB = UB;
    model.neg.pi = fai;
    model.neg.cesi_A = rol1;
    model.neg.cesi_B = rol2;
    model.neg.inv_A = invF1TF1;
    model.neg.inv_B = invF2TF2;
%     model.neg.Z = Z_n;
    
    time = toc;
    % store the model's parameters
    model.name = 'Multi-view Twin SVM';
    model.epsilon = epsilon;
    model.cp = Box_Cons_p;          % positive plane's penalty parameter
    model.cn = Box_Cons_n;          % negative plane's penalty parameter
    model.XAp = XAp;                % viewA's positive train sample
    model.XAn = XAn;                % viewA's negative train sample
    model.XBp = XBp;                % viewB's positive train sample
    model.XBn = XBn;                % viewB's negative train sample
    model.kerneltypeA = kerneltypeA;        % viewA's kernelfunction type
    model.kernelparamA = kernelparamA;      % viewA's kernelfunction's parameter
    model.kerneltypeB = kerneltypeB;        % viewB's kernelfunction type
    model.kernelparamB = kernelparamB;      % viewB's kernelfunction's parameter
    model.CA = CA;                  % viewA's total train sample
    model.CB = CB;                  % viewB's total train sample
    model.uAp = uAp;
    model.uBp = uBp;
    model.vAn = vAn;
    model.vBn = vBn;
    model.training_time = time;     % training time
%     model.training_time = result_p.runtime + result_n.runtime;
    model.predict = @MvTwsvmpredict;    % name of the predict function
    model.num_accuracy = num_accuracy;
    model.num_truncation = num_truncation;
    
    model.testXA = [];
    model.testXB = [];
    model.testy = [];

end