function [ model] = svm2ktrain_gurobi( train_set1, train_set2, train_cat1, D, Box_Cons, type1, sigma1, type2, sigma2, ebsilon)
%   This function is used to train a svm2k model.
% Input:
%   train_set1: viewA's train set with n-by-p, p is the number of attributes;
%   train_set2: viewB's train set with n-by-q, q is the number of attributes;
%   train_cat1: viewA's train label;
%   train_cat2: viewB's train label;
%   D: L1 norm constraint's penalty parameter;
%   Box_Cons: viewA/B's slack penalty vector with 2-by-1;
%   type1: viewA's kernel type, type1 = 'linear', 'rbf', 'poly';
%   sigma1: viewA's kernel parameter;
%   type2: viewB's kernel type, type2 = 'linear', 'rbf', 'poly';
%   sigma2: viewB's kernel parameter;
% Output:
%   model: a structure with decision hyperplane.
% example:
%   [ model] = svm2ktrain( X1, X2, y, D, Box_Cons, type1, sigma1, type2, sigma2, ebsilon)
% Notion:
%   在这个SVM-2K训练中，我们使用了“样本增广”（暂时如此称呼），即给原来的样本尾部加1，x -> [x, 1]
%   令X = [x, 1], W = [w, b];
%   如此一来，w'x + b = <w, x> + b <==> [w, b]'[x, 1]，
%   而ker(X, X) = ker(x, x) + 1;

% - Jiayi Zhu 19th August 2020
% - update by Jiayi Zhu, 26th September 2020
% - update by Jiayi Zhu, 15th September 2021

num_truncation = 8;
global global_options error

%====== main function =====================================================
if nargin < 10
    ebsilon = 0;
end
%rename the train data
tic
X = train_set1;         % viewA's data
X2 = train_set2;        % viewB's data
y = train_cat1;         % sample's label
%     y2 = train_cat1;
Ca = Box_Cons(1, :);
Cb = Box_Cons(2, :);
mm = size(X, 1);        % obtain the number of train set.

E_mat = eye(mm);
O_mat = zeros(mm, mm);
I_mod = ones(mm, 1);
%     O_mod = zeros(mm, 1);

%calculate the kernel matrix
GA = kernelfunction(type1, X, X, sigma1) + 1;     % viewA's [z, 1]' * [z, 1] kernel matrix
GB = kernelfunction(type2, X2, X2, sigma2) + 1;     % viewB's kernel matrix
Y = diag(y);

U = [Y, O_mat, - E_mat, E_mat; O_mat, Y, E_mat, - E_mat];

%% quadprog's data structure
% objective
G = [Y * GA * Y, O_mat, - Y * GA, Y * GA;
    O_mat, Y * GB * Y, Y * GB, - Y * GB;
    - GA * Y, GB * Y, GA + GB, - GA - GB;
    GA * Y, - GB * Y, - GA - GB, GA + GB];
I = [- ones(2*mm, 1);  ebsilon * ones(2*mm, 1)]; % adding ebsilon to this model

% ineq_cons
A = [O_mat O_mat E_mat E_mat];
B = D * I_mod;

% eq_cons
Aeq = [];
Beq = [];

% box bound
LB = zeros(4*mm, 1);
UB = [Ca * I_mod; Cb * I_mod; D * ones(2 * mm, 1)];

%% gurobi's data structure
% objective
model_g.Q = 0.5 * sparse(G);
model_g.obj = I;

% ineq_cons
model_g.A = sparse(A);
model_g.rhs = B;
model_g.sense = '<';

% box bound
model_g.lb = LB;
model_g.ub = UB;

% variable type:(c)ontinuous
model_g.vtype = 'C';
% goal: minimization
model_g.modelsense = 'min';

% solve it
try
    result = gurobi(model_g);
    alpha = result.x;
    alpha = round(alpha, 8);
catch
    [ alpha ] = quadprog(G, I, A, B, Aeq, Beq, LB, UB, [], global_options);
    alpha = round(alpha, 8);
end
%calculate the combined lagrangian multipler
alpha_a = alpha(1 : mm, :);
alpha_b = alpha(mm + 1 : 2 * mm, :);
beta_p = alpha(2 * mm + 1 : 3 * mm, :);
beta_n = alpha(3 * mm + 1 : end, :);
g = U * alpha;
ga = g(1:mm, :);
gb = g((mm + 1): end, :);

%find the support vector, make attention to this place.
num_accuracy = 1e-9;
%     num_accuracy = 0;
%     index_a = find(alpha_a > 0 + num_accuracy & alpha_a < Ca - num_accuracy);
%     index_b = find(alpha_b > 0 + num_accuracy & alpha_b < Cb - num_accuracy);
index_a = find(ga ~= 0);
index_b = find(gb ~= 0);
index_beta_in_epsilon = find(beta_p + beta_n <= num_accuracy);  % 指的是|fA - fB| < epsilon的变量
index_beta_eq_epsilon = find(beta_p + beta_n > 0 + num_accuracy & beta_p + beta_n < D - num_accuracy);  % 指的是|fA - fB| = epsilon的变量
index_beta_out_epsilon = find(beta_p + beta_n >= D - num_accuracy);  % 指的是|fA - fB| > epsilon的变量
%support vector
sv_a = train_set1(index_a, :);
sv_b = train_set2(index_b, :);
%support vector's category
sv_cat1 = y(index_a, :);
sv_cat2 = y(index_b, :);

time = toc;
%output the model
model.name = 'SVM-2K';
% train set
model.CA = X;
model.CB = X2;
model.y = y;
% hyper parameters
model.U = U;
%     model.Box_Cons = Box_Cons;
%     model.D = D;
model.kerneltypeA = type1;
model.kernelparamA = sigma1;
model.kerneltypeB = type2;
model.kernelparamB = sigma2;
model.epsilon = ebsilon;

% QP model
model.obj_quad = G;
model.obj_linear = I;
model.A = A;
model.B = B;
model.UB = UB;
model.Ca = Box_Cons(1, 1);
model.Cb = Box_Cons(2, 1);
model.D  = D;
% todo: model.solver_opt = 已采用全局变量设置global_options

% support vector's index
model.index_a = index_a;
model.index_b = index_b;

% epsilon vector's index
model.index_beta_in_epsilon = index_beta_in_epsilon;
model.index_beta_eq_epsilon = index_beta_eq_epsilon;
model.index_beta_out_epsilon = index_beta_out_epsilon;

% lagrangian multipliers
model.alpha = alpha;
% model.obj_value = result.objval;
model.alpha_A = alpha_a;
model.alpha_B = alpha_b;
model.beta_p = beta_p;
model.beta_n = beta_n;
model.ga = ga;
model.gb = gb;

% supprot vectors
model.support_vec_a = sv_a;
model.support_vec_b = sv_b;
model.support_y1 = sv_cat1;
model.support_y2 = sv_cat2;
%     model.training_time = time;
try
    model.training_time = result.runtime;
catch
    model.training_time = time;
end
% predict function
model.norm_wA = ga' * GA * ga;
model.norm_wB = gb' * GB * gb;
model.predict = @svm2kpredict;
model.testXA = [];
model.testXB = [];
model.testy = [];

clearvars -except model
end
