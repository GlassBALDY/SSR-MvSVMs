function [model_next, detailed] = DVI_rules_4_SVM2K(svm2kmodel, C_next, opts, error)
%   This function is used to implement a SVM2K-DVI framework
% Input:
%   obj_quad: global quadratic matrix;
%   X: training data;
%   y: training label;
%   C_next: t step's penalty parameter;
%   C_pore: t-1 step's penalty parameter;
%   epsilon: disimilarity parameter;
%   pi_pre: t-1 step's optimal solution;
% Output:
%   pi_next: t step's lagrangian multiplier.
% Usage:
%   [model_next, detailed] = DVI_rules_4_SVM2K(svm2kmodel, C_next, opts)

% - Jiayi Zhu, 2021.07.08

% ====== main function ====================================================
% import the previous svm2k model
pi_pre = svm2kmodel.alpha;
obj_quad = svm2kmodel.obj_quad;
obj_linear = svm2kmodel.obj_linear;
A = svm2kmodel.A;
%     B = svm2kmodel.B;
C_pre = svm2kmodel.UB;
epsilon = svm2kmodel.epsilon;

% 生成 C_next {分两种情况： C_next传入scalar、C_next传入array}
L = size(obj_quad, 1); l = L/4;
if isscalar(C_next)     % C_next传入scalar
    Ca = C_next; Cb = C_next; D = C_next;
    C_next = C_next * ones(L, 1);
else                    % C_next传入array
    Ca = C_next(1); Cb = C_next(2); D = C_next(3);
    C_next = C_next * [ones(1, l), zeros(1, l), zeros(1, l), zeros(1, l);
        zeros(1, l), ones(1, l), zeros(1, l), zeros(1, l);
        zeros(1, l), zeros(1, l), ones(1, l), ones(1, l)];
    C_next = C_next';
end



%% Screening (DVI-SVM2K) 选出非支持向量并去掉
timer = tic;    % 计时
% 如果前后C_series无区别，直接跳过
if C_pre == C_next
    jump_time = toc(timer);
    model_next = svm2kmodel;
    model_next.training_time = jump_time;
    detailed.Num_notraining = L;
    return
end

%% mini functions
PpP0_pi = @(pi) (C_next ./ C_pre + 1) .* pi;
PmP0_pi = @(pi) (C_next ./ C_pre - 1) .* pi;

ZZT_PpP0_pi = 0.5 * (obj_quad * PpP0_pi(pi_pre));
norm_Z_norm_Z_PmP0_pi = 0.5 * sqrt(diag(obj_quad)) * sqrt(PmP0_pi(pi_pre)' * (obj_quad * (PmP0_pi(pi_pre))));

% calculate error bound
% PpP0 = diag(C_next ./ C_pre) + speye(L); PmP0 = diag(C_next ./ C_pre) - speye(L);
% [~, y] = eig(PmP0 * obj_quad * PmP0);
% norm_ZT_PmP0 = sqrt(max(diag(y)));
% errors = 0.5 * (sqrt(diag(obj_quad)) .* norm_ZT_PmP0 + sqrt(sum((obj_quad * PpP0) .^ 2, 2))) * error;

% a casual setting:
errors = 1e-10;

% calculate SSR 1
SSR_upper = ZZT_PpP0_pi + norm_Z_norm_Z_PmP0_pi + errors;

% calculate SSR 2
SSR_lower = ZZT_PpP0_pi - norm_Z_norm_Z_PmP0_pi - errors;

%     over_P0_pi = @(pi) pi ./ C_pre;
%     ZZT_pi0 = obj_quad * pi_pre;
%     ZZT_over_P0_pi0 = obj_quad * over_P0_pi(pi_pre);
%     norm_ZZT_over_P0_pi0 = sqrt(diag(obj_quad)) * sqrt(PmP0_pi(pi_pre)' * obj_quad * PmP0_pi(pi_pre));
%
%     % 1. alpha_A/B
%     criteria_alpha_0 = (2 * (1 - ZZT_pi0(1: 2*l, :))) ./ ...
%         (ZZT_over_P0_pi0(1: 2*l, :) - norm_ZZT_over_P0_pi0(1: 2*l, :));
%     d_alpha_0 = prctile(criteria_alpha_0,70);
%
%     criteria_alpha_C = (2 * (1 - ZZT_pi0(1: 2*l, :))) ./ ...
%         (ZZT_over_P0_pi0(1: 2*l, :) + norm_ZZT_over_P0_pi0(1: 2*l, :));
%     d_alpha_C = prctile(criteria_alpha_C,30);
%
%     % 2.1 beta_p +
%     criteria_beta_p_0 = (2 * (epsilon - 2 * ZZT_pi0(2*l+1 : 3*l, :))) ./ ...
%         (ZZT_over_P0_pi0(2*l+1 : 3*l, :) + norm_ZZT_over_P0_pi0(2*l+1 : 3*l, :));
%     d_beta_p_0 = prctile(criteria_beta_p_0,30);
%
%     criteria_beta_p_D = (2 * (epsilon - 2 * ZZT_pi0(2*l+1 : 3*l, :))) ./ ...
%         (ZZT_over_P0_pi0(2*l+1 : 3*l, :) - norm_ZZT_over_P0_pi0(2*l+1 : 3*l, :));
%     d_beta_p_D = prctile(criteria_beta_p_D,30);
%
%     % 2.2 beta_n -
%     criteria_beta_n_0 = (2 * (epsilon - 2 * ZZT_pi0(3*l+1 : end, :))) ./ ...
%         (ZZT_over_P0_pi0(3*l+1 : end, :) + norm_ZZT_over_P0_pi0(3*l+1 : end, :));
%     d_beta_n_0 = prctile(criteria_beta_n_0,30);
%
%     criteria_beta_n_D = (2 * (epsilon - 2 * ZZT_pi0(3*l+1 : end, :))) ./ ...
%         (ZZT_over_P0_pi0(3*l+1 : end, :) - norm_ZZT_over_P0_pi0(3*l+1 : end, :));
%     d_beta_n_D = prctile(criteria_beta_n_D,30);

% 安全样本筛选规则与二次矩阵的顺序有关，一定注意SSR规则要和相观的矩阵子块对应！
% safe screening
idx_round = 1:l; num_accuracy = 1e-8;
% screening alpha_A
idx_alphaA_0 = find(SSR_lower(idx_round, :) > 1);
idx_alphaA_C = find(SSR_upper(idx_round, :) < 1);
% idx_alphaA_un = find(SSR_upper(idx_round, :) >= 1 - num_accuracy & SSR_lower(idx_round, :) <= 1 + num_accuracy);
idx_alphaA_un = find(SSR_upper(idx_round, :) >= 1 & SSR_lower(idx_round, :) <= 1);
% screening alpha_B
idx_alphaB_0 = find(SSR_lower(l + idx_round, :) > 1) + l;
idx_alphaB_C = find(SSR_upper(l + idx_round, :) < 1) + l;
idx_alphaB_un = find(SSR_upper(l + idx_round, :) >= 1 & SSR_lower(l + idx_round, :) <= 1) + l;

% % screening beta +
% idx_betap_0 = find(SSR_upper(2 * l + idx_round, :) < epsilon) + 2 * l;
% idx_betap_D = find(SSR_lower(2 * l + idx_round, :) > epsilon) + 2 * l;
% idx_betap_un = find(SSR_upper(2 * l + idx_round, :) >= epsilon & SSR_lower(2 * l + idx_round, :) <= epsilon) + 2 * l;
% 
% % screening beta -
% idx_betan_0 = find(SSR_upper(3 * l + idx_round, :) < epsilon) + 3 * l;
% idx_betan_D = find(SSR_lower(3 * l + idx_round, :) > epsilon) + 3 * l;
% idx_betan_un = find(SSR_lower(3 * l + idx_round, :) <= epsilon & SSR_upper(3 * l + idx_round, :) >= epsilon) + 3 * l;

% try:
% screening beta +
idx_betapn_0 = find(SSR_upper(3 * l + idx_round, :) < epsilon) + 2 * l;
idx_betap_D = find(SSR_lower(3 * l + idx_round, :) > epsilon) + 2 * l;
idx_betap_un = find(SSR_upper(3 * l + idx_round, :) >= epsilon & SSR_lower(3 * l + idx_round, :) <= epsilon) + 2 * l;

% screening beta -
idx_betanp_0 = find(SSR_upper(2 * l + idx_round, :) < epsilon) + 3 * l;
idx_betan_D = find(SSR_lower(2 * l + idx_round, :) > epsilon) + 3 * l;
idx_betan_un = find(SSR_lower(2 * l + idx_round, :) <= epsilon & SSR_upper(2 * l + idx_round, :) >= epsilon) + 3 * l;

%     if Ca == Ca_pre
%         idx_alphaA_0  = find(pi_pre(idx_round, :) < 0 + num_accuracy);
%         idx_alphaA_C  = find(pi_pre(idx_round, :) > Ca_pre - num_accuracy);
%         idx_alphaA_un = [];
%     end
%
%     if Cb == Cb_pre
%         idx_alphaB_0  = find(pi_pre(l + idx_round, :) < 0 + num_accuracy) + l;
%         idx_alphaB_C  = find(pi_pre(l + idx_round, :) > Cb_pre - num_accuracy) + l;
%         idx_alphaB_un = [];
%     end
%
%     if D  == D_pre
%         idx_betapn_0 = find(pi_pre(3 * l + idx_round, :) < 0 + num_accuracy) + 2 * l;
%         idx_betap_D  = find(pi_pre(3 * l + idx_round, :) > D_pre - num_accuracy) + 2 * l;
%         idx_betap_un = [];
%
%         idx_betanp_0 = find(pi_pre(2 * l + idx_round, :) < 0 + num_accuracy) + 3 * l;
%         idx_betan_D  = find(pi_pre(2 * l + idx_round, :) > D_pre - num_accuracy) + 3 * l;
%         idx_betan_un = [];
%     end

idx_alpha_0 =  [idx_alphaA_0; idx_alphaB_0; idx_betapn_0; idx_betanp_0];
idx_alpha_un = [idx_alphaA_un; idx_alphaB_un; idx_betap_un; idx_betan_un];
idx_alpha_C =  [idx_alphaA_C; idx_alphaB_C; idx_betap_D; idx_betan_D];

% 二次规划二次项矩阵 （取乘子未知的部分组成）
obj_quad_next = obj_quad(idx_alpha_un, idx_alpha_un);
% 二次规划线性部分 （注意保留一次项）
if isempty(idx_alpha_C) == 1 && isempty(idx_alpha_0) == 1
    % 当集合L为空且集合R为空时，继承model_pre的线性项
    obj_linear_next = obj_linear;
elseif isempty(idx_alpha_C) == 1 && isempty(idx_alpha_0) ~= 1
    % 当集合L为空且集合R不为空时，取乘子未知的部分组成新的线性项
    obj_linear_next = obj_linear(idx_alpha_un, :);
else
    % 当集合L不为空时，取乘子未知的部分，与集合L的交叉项组成新的线性项
    obj_linear_next = obj_linear(idx_alpha_un, :) + obj_quad(idx_alpha_un, idx_alpha_C) * C_next(idx_alpha_C, :); % to do
end
%     I = - ones(U, 1); % to do

% 存在不等式约束！不等式约束如何处理？
% 不等式约束应该重新生成
% ineq_cons
A_next = A(idx_betap_un - 2 * l, idx_alpha_un);
B = D * ones(l, 1);
B_next = B(idx_betap_un - 2 * l, 1);
% 上界与下界
lb = zeros(size(idx_alpha_un, 1), 1);%变量限制，变量theta的上界
ub = C_next(idx_alpha_un, :);

% 迭代次数，停机准则都设置一下
% opts = optimset('Algorithm','interior-point-convex','Display','off', 'TolCon', 1e-10, 'MaxIter', 1000);
% 求解
if isempty(obj_quad_next) ~= 1
    pi_temp = quadprog(obj_quad_next, obj_linear_next, A_next, B_next, [], [], lb, ub, [], opts);%求解二次规划问题,求出theta1
    pi_temp = round(pi_temp, 8);
    %     pi_temp = quadprog(obj_quad_next, obj_linear_next, A_next, B_next, [], [], lb, ub, []);%求解二次规划问题,求出theta1
else
    pi_temp = [];
end
% 输出pi_next
pi_next = pi_pre;
pi_next(idx_alpha_un, :) = pi_temp;

%     if Ca ~= Ca_pre
pi_next(idx_alphaA_0, :) = 0;
pi_next(idx_alphaA_C, :) = C_next(idx_alphaA_C, :);
%     end
%
%     if Cb ~= Cb_pre
pi_next(idx_alphaB_0, :) = 0;
pi_next(idx_alphaB_C, :) = C_next(idx_alphaB_C, :);
%     end
%
%     if D  ~= D_pre
pi_next(idx_betapn_0, :) = 0;
pi_next(idx_betap_D, :) = C_next(idx_betap_D, :);

pi_next(idx_betanp_0, :) = 0;
pi_next(idx_betan_D, :) = C_next(idx_betan_D, :);
pi_next = round(pi_next, 8);
%     end

%calculate the combined lagrangian multipler
alpha_a = pi_next(1 : l, :);
alpha_b = pi_next(l + 1 : 2 * l, :);
beta_p = pi_next(2 * l + 1 : 3 * l, :);
beta_n = pi_next(3 * l + 1 : end, :);
U = svm2kmodel.U;
g = U * pi_next;
ga = g(1:l, :);
gb = g((l + 1): end, :);

SSR_time = toc(timer);

%find the support vector, make attention to this place.
% 注意SVM-2K所谓支撑向量这一说法，在多视角下，如何界定支撑向量？
% 按照MvNPSVM的做法，应该是取 综合系数 ~= 0 的部分作为支撑向量？
%     index_a = find(alpha_a > 0 + num_accuracy & alpha_a < Ca - num_accuracy);
%     index_b = find(alpha_b > 0 + num_accuracy & alpha_b < Cb - num_accuracy);

index_a = find(ga ~= 0);
index_b = find(gb ~= 0);
index_beta_in_epsilon = find(beta_p + beta_n <= num_accuracy);  % 指的是|fA - fB| < epsilon的变量
index_beta_eq_epsilon = find(beta_p + beta_n > 0 + num_accuracy & beta_p + beta_n < D - num_accuracy);  % 指的是|fA - fB| = epsilon的变量
index_beta_out_epsilon = find(beta_p + beta_n >= D - num_accuracy);  % 指的是|fA - fB| > epsilon的变量
%support vector
train_set1 = svm2kmodel.CA; train_set2 = svm2kmodel.CB; y = svm2kmodel.y;
sv_a = train_set1(index_a, :);
sv_b = train_set2(index_b, :);
%support vector's category
sv_cat1 = y(index_a, :);
sv_cat2 = y(index_b, :);

%output the model
model_next = svm2kmodel;

model_next.Ca = Ca;
model_next.Cb = Cb;
model_next.D  = D;
model_next.UB = C_next;     % you need to update the upper bound !!!
% support vector's index

model_next.index_a = index_a;
model_next.index_b = index_b;

% epsilon vector's index
model_next.index_beta_in_epsilon = index_beta_in_epsilon;
model_next.index_beta_eq_epsilon = index_beta_eq_epsilon;
model_next.index_beta_out_epsilon = index_beta_out_epsilon;

% lagrangian multipliers
model_next.alpha = pi_next;
model_next.alpha_A = alpha_a;
model_next.alpha_B = alpha_b;
model_next.beta_p = beta_p;
model_next.beta_n = beta_n;
model_next.ga = ga;
model_next.gb = gb;
model_next.training_time = SSR_time;

% supprot vectors
model_next.support_vec_a = sv_a;
model_next.support_vec_b = sv_b;
model_next.support_y1 = sv_cat1;
model_next.support_y2 = sv_cat2;

% norm
%     model_next.norm_wA = ga' * GA * ga;
%     model_next.norm_wB = gb' * GB * gb;

% SSR_detail
% screening alpha_A
detailed.idx_alphaA_0 = idx_alphaA_0;
detailed.idx_alphaA_C = idx_alphaA_C;
detailed.idx_alphaA_un = idx_alphaA_un;
% screening alpha_B
detailed.idx_alphaB_0 = idx_alphaB_0 - l;
detailed.idx_alphaB_C = idx_alphaB_C - l;
detailed.idx_alphaB_un = idx_alphaB_un - l;
% screening beta +
detailed.idx_betapn_0 = idx_betapn_0 - 2 * l;
detailed.idx_betap_D = idx_betap_D - 2 * l;
detailed.idx_betap_un = idx_betap_un - 2 * l;
% screening beta -
detailed.idx_betanp_0 = idx_betanp_0 - 3 * l;
detailed.idx_betan_D = idx_betan_D - 3 * l;
detailed.idx_betan_un = idx_betan_un - 3 * l;
% total
detailed.idx_alpha_un = idx_alpha_un;
detailed.idx_alpha_C = idx_alpha_C;
% SSR rules
detailed.SSR1 = SSR_upper;
detailed.SSR2 = SSR_lower;
detailed.screened_number = L - size(idx_alpha_un, 1);
model_next.SSR_detail = detailed;

clearvars -except model_next detailed
end

