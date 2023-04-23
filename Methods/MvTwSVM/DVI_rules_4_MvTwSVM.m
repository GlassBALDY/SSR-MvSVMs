function [model_next, detailed] = DVI_rules_4_MvTwSVM(MvTwSVMmodel, C_next_pos, C_next_neg, opts)
%   This function is used to implement a SVM2K-DVI framework
% Input:
%   MvTwSVMmodel: the multi-view Twin SVMs trained with C0;
%   C_next_pos: positive problem's C1;
%   C_next_neg: negative problem's C1;
%   opts: solver's options;
% // Done: change C_next to a vector.
% Output:
%   model_next: the multi-view Twin SVMs trained with C1;
%   detailed: records related to SSR.
% Usage:
%   [mvtwsvmmodel_next, detailed] = DVI_rules_4_MvTwSVM(mvtwsvmmodel, C_next_pos, C_next_neg, opts);
%
% - Jiayi Zhu, 2021.07.08
% - Jiayi Zhu, 2021.10.23

% ====== main function ====================================================
    % import the previous svm2k model
    timer = tic;    % ��ʱ
    % ����ӦC1����������ô���
    % ɸѡ + ģ��
    [MvTwSVMmodel, detailed_pos] = sub_DVI_rules_4_MvTwSVM_pos(MvTwSVMmodel, C_next_pos, opts);
    % ɸѡ - ģ��
    [model_next,   detailed_neg] = sub_DVI_rules_4_MvTwSVM_neg(MvTwSVMmodel, C_next_neg, opts);
    SSR_time = toc(timer);
    
    % update model
    model_next.cp = C_next_pos;
    model_next.cn = C_next_neg;
    model_next.training_time = SSR_time;
    
    % record the details of SSR
    detailed.detailed_pos = detailed_pos;
    detailed.detailed_neg = detailed_neg;
end

function [model_next, detailed] = sub_DVI_rules_4_MvTwSVM_pos(model, C_next, opts)
    %% ��ģ��ɸѡ���� 
    % ���룺
    % - model = C0�µ�ģ��
    % - C_next = �ͷ�����C1������������[Ca, Cb, D]��Ҳ�����ǳ���C
    % - opts = �������
    % �����
    % - model_next = C1�µ�ģ��
    % - detailed = SSRɸѡ���̼�¼
    
    num_accuracy = model.num_accuracy; 
    num_truncation = model.num_truncation;
    
    % ================ main function ======================================
    % �ؼ��������룺
    % C0ʱ���Ž�
    pi_pre = model.pos.pi;
    % MvTwSVM + ģ�� ��ż�Ķ��ξ���һ�������ʽԼ��������Լ���Ͻ�
    obj_quad = model.pos.obj_quad;
    obj_linear = model.pos.obj_linear;
    A = model.pos.A;
    B = model.pos.B;
    C_pre = model.pos.UB;
    % ������Լ���� epsilon ������ϵ��
    epsilon = model.epsilon;
    
    %% Screening (DVI-MvTwSVM) Ԥɸѡ�������ճ���
    % ���� + ���������� - ��������
    l_p = size(model.XAp, 1);
    l_n = size(model.XAn, 1);
    
    % ���� C_next {����������� C_next����scalar��C_next����array}
    L = size(obj_quad, 1); 
    if isscalar(C_next) == 1    % C_next����scalar
        Ca = C_next; Cb = C_next; D = C_next;
        C_next = C_next * ones(L, 1);
    else                        % C_next����array
        Ca = C_next(1); Cb = C_next(2); D = C_next(3);
        C_next = C_next * [ones(1, l_n), zeros(1, l_n), zeros(1, l_p), zeros(1, l_p); 
                           zeros(1, l_n), ones(1, l_n), zeros(1, l_p), zeros(1, l_p); 
                           zeros(1, l_n), zeros(1, l_n), ones(1, l_p), ones(1, l_p)];
        C_next = C_next';
    end
    % �����۴���ʲô��C_next��������һ�� L-by-1 ��������
    
    % ��� C0 = C1������ȥɸѡ��ֱ����C0�Ľ���ΪC1�Ľ�
    if C_pre == C_next
        jump_time = toc(timer);
        model_next = model;
        model_next.training_time = jump_time;
        detailed.Num_notraining = L;
        return
    end
    
%     C_next = C_next * ones(L, 1);
    % ����ɸѡ׼��
    PpP0 = (C_next ./ C_pre) + 1; PmP0 = (C_next ./ C_pre) - 1;
    M = 0.5 * obj_quad * (PpP0 .* pi_pre);
%     N = 0.5 * sqrt(sum(([model.pos.cesi_A; model.pos.cesi_B]' / model.pos.Z) .^ 2, 2)) ...
%         * sqrt((pi_pre' .* PmP0') * obj_quad * (PmP0 .* pi_pre));
%     N = 0.5 * sqrt(sum(([model.pos.cesi_A; model.pos.cesi_B]' / model.pos.Z) .^ 2, 2)) ...
%         * sqrt((pi_pre') * obj_quad * (pi_pre));
    N = 0.5 * sqrt(diag(obj_quad)) * sqrt((pi_pre' .* PmP0') * obj_quad * (PmP0 .* pi_pre));

    % calculate SSR 1
    SSR1 = M + N; 
    % calculate SSR 2
    SSR2 = M - N;
    
    % safe screening
    idx_round_n = 1: l_n;
    idx_round_p = 1: l_p;
    % screening alpha_A
    idx_alphaA_0 = find(SSR2(idx_round_n, :) > 1 + num_accuracy);
    idx_alphaA_C = find(SSR1(idx_round_n, :) < 1 - num_accuracy);
    idx_alphaA_un = find(SSR1(idx_round_n, :) >= 1 & SSR2(idx_round_n, :) <= 1);
    % screening alpha_B
    idx_alphaB_0 = find(SSR2(l_n + idx_round_n, :) > 1 + num_accuracy) + l_n;
    idx_alphaB_C = find(SSR1(l_n + idx_round_n, :) < 1 - num_accuracy) + l_n;
    idx_alphaB_un = find(SSR1(l_n + idx_round_n, :) >= 1 & SSR2(l_n + idx_round_n, :) <= 1) + l_n;
    
    % screening beta +
    idx_betapn_0 = find(SSR1(2 * l_n + l_p + idx_round_p, :) < epsilon - num_accuracy) + 2 * l_n;
    idx_betap_D  = find(SSR2(2 * l_n + l_p + idx_round_p, :) > epsilon + num_accuracy) + 2 * l_n;
    idx_betap_un = find(SSR1(2 * l_n + l_p + idx_round_p, :) >= epsilon & SSR2(2 * l_n + l_p + idx_round_p, :) <= epsilon) + 2 * l_n;
    % screening beta -
    idx_betanp_0 = find(SSR1(2 * l_n + idx_round_p, :) < epsilon - num_accuracy) + 2 * l_n + l_p;
    idx_betan_D  = find(SSR2(2 * l_n + idx_round_p, :) > epsilon + num_accuracy) + 2 * l_n + l_p;
    idx_betan_un = find(SSR2(2 * l_n + idx_round_p, :) <= epsilon & SSR1(2 * l_n + idx_round_p, :) >= epsilon) + 2 * l_n + l_p;
 
    idx_alpha_0 = [idx_alphaA_0; idx_alphaB_0; idx_betapn_0; idx_betanp_0];
    idx_alpha_un = [idx_alphaA_un; idx_alphaB_un; idx_betap_un; idx_betan_un];
    idx_alpha_C = [idx_alphaA_C; idx_alphaB_C; idx_betap_D; idx_betan_D];
    
    %% ���������⣬����ȷ�����ι滮
    % \\ To do
    % ���ι滮���������
    obj_quad_next = obj_quad(idx_alpha_un, idx_alpha_un); 
    % ���ι滮���Բ���
    if isempty(idx_alpha_C) == 1 && isempty(idx_alpha_0) == 1
        obj_linear_next = obj_linear;
    elseif isempty(idx_alpha_C) == 1 && isempty(idx_alpha_0) ~= 1
        obj_linear_next = obj_linear(idx_alpha_un, :);
    else
        obj_linear_next = obj_linear(idx_alpha_un, :) + obj_quad(idx_alpha_un, idx_alpha_C) * C_next(idx_alpha_C, :); % to do
    end
    
    % ����ʽԼ��
    A_next = A(idx_betap_un - 2 * l_n, idx_alpha_un);
    % ��D_next���½綨
    B_next = C_next(idx_betap_un, :);  
%     B = D * ones(l_p, 1);
%     B_next = B(idx_betap_un - 2 * l_n, 1);

    % �Ͻ����½�
    lb = zeros(size(idx_alpha_un, 1), 1);%�������ƣ�����theta���Ͻ�
    ub = C_next(idx_alpha_un, :);
    
    % pi_next����ֵ
    pi_next = zeros(L, 1);
    pi_next(idx_alphaA_0, :) = 0;
    pi_next(idx_alphaA_C, :) = C_next(idx_alphaA_C, :);
    
    pi_next(idx_alphaB_0, :) = 0;
    pi_next(idx_alphaB_C, :) = C_next(idx_alphaB_C, :);
    
    pi_next(idx_betapn_0, :) = 0;
    pi_next(idx_betap_D, :) = C_next(idx_betap_D, :);
    
    pi_next(idx_betanp_0, :) = 0;
    pi_next(idx_betan_D, :) = C_next(idx_betan_D, :);
    
    % ����֤��SSR������ȷ
    
    % ��� reduced_MvTwSVMs
    if isempty(obj_quad_next) ~= 1
        % ���obj_quad_next�ǿգ������reduced_MvTwSVM
        pi_temp = quadprog(obj_quad_next, obj_linear_next, A_next, B_next, [], [], lb, ub, [], opts);
    else
        % �����������
        pi_temp = [];
    end
    
    % �� unknown ��pi��ֵpi_temp
    pi_next(idx_alpha_un, :) = pi_temp;

    % piȫ���������������һ���и��������и�ø��ֲ�
    pi_next = round(pi_next, num_truncation);
    
    % Ϊ����֤Ψһ�����⣬����Ŀ�꺯��ֵ
    obj_value = 0.5 * pi_next' * obj_quad * pi_next + obj_linear' * pi_next;
    
    %calculate the combined lagrangian multipler
    alpha_a = pi_next(idx_round_n, :);
    alpha_b = pi_next(l_n + idx_round_n, :);
    beta_p  = pi_next(2 * l_n + idx_round_p, :);
    beta_n  = pi_next(2 * l_n + l_p + idx_round_p, :);
    
%     index_a = find(alpha_a > 0 + num_accuracy & alpha_a < Ca - num_accuracy);
%     index_b = find(alpha_b > 0 + num_accuracy & alpha_b < Cb - num_accuracy);
   
    % ����Ѱ�Ҷ�Ӧ��֧��������
%     sv_index_a = find(alpha_a > 0 + num_accuracy );
%     sv_index_b = find(alpha_b > 0 + num_accuracy );
    
    % Ѱ������һ����Լ��������
    index_beta_in_epsilon = find(beta_p + beta_n <= num_accuracy);  % ָ����|fA - fB| < epsilon�ı���
    index_beta_eq_epsilon = find(beta_p + beta_n > 0 + num_accuracy & beta_p + beta_n < D - num_accuracy);  % ָ����|fA - fB| = epsilon�ı���
    index_beta_out_epsilon = find(beta_p + beta_n >= D - num_accuracy);  % ָ����|fA - fB| > epsilon�ı���
    
    % ���� model_pre ��������Ͼ������������
    cesi_A = model.pos.cesi_A;
    cesi_B = model.pos.cesi_B;
    inv_A = model.pos.inv_A;
    inv_B = model.pos.inv_B;
    
    % �����ʽ��
    lag_piA_next = cesi_A * pi_next;
    lag_piB_next = cesi_B * pi_next;
    
    % calculate positive decision parameters
    uA = inv_A * lag_piA_next;
    uB = inv_B * lag_piB_next;
   
    % =====================================================================
    
    % output the model ���� model_pre ģ��
    % inherite the previous model
    model_next = model;
    
    % ���� + ģ�͵��Ͻ�
    model_next.pos.UB = C_next;
    % ���� + ģ�͵�Ŀ�꺯��ֵ
    model_next.pos.obj_value = obj_value;

    % update pi 
    model_next.pos.pi = pi_next;
    model_next.pos.alpha_a = alpha_a;
    model_next.pos.alpha_b = alpha_b;
    model_next.pos.beta_p = beta_p;
    model_next.pos.beta_n = beta_n;
    
    % update positive decision parameters
    model_next.uAp = uA;
    model_next.uBp = uB;
    
    model_next.Ca = Ca;
    model_next.Cb = Cb;
    model_next.D  = D;
    
%     % support vector's index
%     model_next.pos.index_a = sv_index_a;
%     model_next.pos.index_b = sv_index_b;
    
    % epsilon vector's index
    model_next.pos.index_beta_in_epsilon  = index_beta_in_epsilon;
    model_next.pos.index_beta_eq_epsilon  = index_beta_eq_epsilon;
    model_next.pos.index_beta_out_epsilon = index_beta_out_epsilon;
    
    % ==== SSR record =====================================================
    SSR_indicator = ones(size(pi_next));
    SSR_indicator(idx_alpha_0, :) = 0;
    SSR_indicator(idx_alpha_C, :) = 1;
    SSR_indicator(idx_alpha_un, :) = 1/2;
    
    % screening alpha_A
    detailed.idx_alphaA_0  = idx_alphaA_0;
    detailed.idx_alphaA_C  = idx_alphaA_C;
    detailed.idx_alphaA_un = idx_alphaA_un;
    
    % screening alpha_B
    detailed.idx_alphaB_0  = idx_alphaB_0;
    detailed.idx_alphaB_C  = idx_alphaB_C;
    detailed.idx_alphaB_un = idx_alphaB_un;
    
    % screening beta +
    detailed.idx_betapn_0 = idx_betapn_0;
    detailed.idx_betap_D  = idx_betap_D;
    detailed.idx_betap_un = idx_betap_un;
    
    % screening beta -
    detailed.idx_betanp_0 = idx_betanp_0;
    detailed.idx_betan_D  = idx_betan_D;
    detailed.idx_betan_un = idx_betan_un;
    
    % total
    detailed.idx_alpha_0  = idx_alpha_0;
    detailed.idx_alpha_C  = idx_alpha_C;
    detailed.idx_alpha_un = idx_alpha_un;
    
    % SSR rules
    detailed.SSR1 = SSR1;
    detailed.SSR2 = SSR2;
    detailed.SSR_indicator = SSR_indicator;
    detailed.screened_number = L - size(idx_alpha_un, 1);
    model_next.pos.SSR_detail = detailed;
    clearvars -except model_next detailed
end

function [model_next, detailed] = sub_DVI_rules_4_MvTwSVM_neg(model, C_next, opts)
    num_accuracy = model.num_accuracy; 
    num_truncation = model.num_truncation;
    
    % ================ main function ======================================
    pi_pre = model.neg.pi;
    obj_quad = model.neg.obj_quad;
    obj_linear = model.neg.obj_linear;
    A = model.neg.A;
%     B = model.neg.B;
    C_pre = model.neg.UB;
    epsilon = model.epsilon;

    %% Screening (DVI-MvTwSVM) ѡ����֧��������ȥ��
    L = size(obj_quad, 1); 
    l_p = size(model.XAp, 1);
    l_n = size(model.XAn, 1);
    % ���� C_next {����������� C_next����scalar��C_next����array}
    if isscalar(C_next) == 1    % C_next����scalar
        Ca = C_next; Cb = C_next; D = C_next;
        C_next = C_next * ones(L, 1);
    else                        % C_next����array
        Ca = C_next(1); Cb = C_next(2); D = C_next(3);
        C_next = C_next * [ones(1, l_p), zeros(1, l_p), zeros(1, l_n), zeros(1, l_n); 
                           zeros(1, l_p), ones(1, l_p), zeros(1, l_n), zeros(1, l_n); 
                           zeros(1, l_p), zeros(1, l_p), ones(1, l_n), ones(1, l_n)];
        C_next = C_next';
    end
    
    % ���ǰ��C_series������ֱ������
    if C_pre == C_next
        jump_time = toc(timer);
        model_next = model;
        model_next.training_time = jump_time;
        detailed.Num_notraining = L;
        return
    end
    
    PpP0 = (C_next ./ C_pre) + 1; PmP0 = (C_next ./ C_pre) - 1;
    M = 0.5 * obj_quad * (PpP0 .* pi_pre);
%     N = 0.5 * sqrt(sum(([model.neg.cesi_A; model.neg.cesi_B]' / model.neg.Z) .^ 2, 2)) ...
%         * sqrt((pi_pre' .* PmP0') * obj_quad * (PmP0 .* pi_pre));
    N = 0.5 * sqrt(diag(obj_quad)) * sqrt((pi_pre' .* PmP0') * obj_quad * (PmP0 .* pi_pre));
%     N = 0.5 * sqrt(sum(([model.neg.cesi_A; model.neg.cesi_B]' / model.neg.Z) .^ 2, 2)) ...
%         * sqrt(pi_pre' * obj_quad * pi_pre);
    % calculate SSR 1
    SSR1 = M + N; 
    % calculate SSR 2
    SSR2 = M - N;
    
    % safe screening
    idx_round_n = 1: l_n;
    idx_round_p = 1: l_p; 
    % screening alpha_A
    idx_alphaA_0  = find(SSR2(idx_round_p, :) > 1 + num_accuracy);
    idx_alphaA_C  = find(SSR1(idx_round_p, :) < 1 - num_accuracy);
    idx_alphaA_un = find(SSR1(idx_round_p, :) >= 1 & SSR2(idx_round_p, :) <= 1);
    % screening alpha_B
    idx_alphaB_0  = find(SSR2(l_p + idx_round_p, :) > 1 + num_accuracy) + l_p;
    idx_alphaB_C  = find(SSR1(l_p + idx_round_p, :) < 1 - num_accuracy) + l_p;
    idx_alphaB_un = find(SSR1(l_p + idx_round_p, :) >= 1 & SSR2(l_p + idx_round_p, :) <= 1) + l_p;
    % screening beta +
    idx_betapn_0 = find(SSR1(2 * l_p + l_n + idx_round_n, :) < epsilon - num_accuracy) + 2 * l_p;
    idx_betap_D  = find(SSR2(2 * l_p + l_n + idx_round_n, :) > epsilon + num_accuracy) + 2 * l_p;
    idx_betap_un = find(SSR1(2 * l_p + l_n + idx_round_n, :) >= epsilon & SSR2(2 * l_p + l_n + idx_round_n, :) <= epsilon) + 2 * l_p;
    % screening beta -
    idx_betanp_0 = find(SSR1(2 * l_p + idx_round_n, :) < epsilon - num_accuracy) + 2 * l_p + l_n;
    idx_betan_D  = find(SSR2(2 * l_p + idx_round_n, :) > epsilon + num_accuracy) + 2 * l_p + l_n;
    idx_betan_un = find(SSR1(2 * l_p + idx_round_n, :) >= epsilon & SSR2(2 * l_p + idx_round_n, :) <= epsilon) + 2 * l_p + l_n;
    
    idx_alpha_0 = [idx_alphaA_0; idx_alphaB_0; idx_betapn_0; idx_betanp_0];
    idx_alpha_un = [idx_alphaA_un; idx_alphaB_un; idx_betap_un; idx_betan_un];
    idx_alpha_C = [idx_alphaA_C; idx_alphaB_C; idx_betap_D; idx_betan_D];
    
    % ���ι滮���������
    obj_quad_next = obj_quad(idx_alpha_un, idx_alpha_un); 
    % ���ι滮���Բ���
    if isempty(idx_alpha_C) == 1 && isempty(idx_alpha_0) == 1
        obj_linear_next = obj_linear;
    elseif isempty(idx_alpha_C) == 1 && isempty(idx_alpha_0) ~= 1
        obj_linear_next = obj_linear(idx_alpha_un, :);
    else
        obj_linear_next = obj_linear(idx_alpha_un, :) + obj_quad(idx_alpha_un, idx_alpha_C) * C_next(idx_alpha_C, :); % to do
    end
    % ineq_cons
    A_next = A(idx_betap_un - 2 * l_p, idx_alpha_un);   
    B = D * ones(l_n, 1);
    B_next = B(idx_betap_un - 2 * l_p, 1);  
    % �Ͻ����½�
    lb = zeros(size(idx_alpha_un, 1), 1);%�������ƣ�����theta���Ͻ�
    ub = C_next(idx_alpha_un, :);
    
    % ��������pi_next��ֵ�ɵ�pi_pre
    pi_next = pi_pre;
    
    % ��ɸѡ�ĳ����ȸ�ֵ
    pi_next(idx_alphaA_0, :) = 0;
    pi_next(idx_alphaA_C, :) = C_next(idx_alphaA_C, :);
    
    pi_next(idx_alphaB_0, :) = 0;
    pi_next(idx_alphaB_C, :) = C_next(idx_alphaB_C, :);
    
    pi_next(idx_betapn_0, :) = 0;
    pi_next(idx_betap_D, :) = C_next(idx_betap_D, :);
    
    pi_next(idx_betanp_0, :) = 0;
    pi_next(idx_betan_D, :) = C_next(idx_betan_D, :);
    
    % opts = optimset('Algorithm','interior-point-convex','Display','off', 'TolCon', 1e-10, 'MaxIter', 1000);
    if isempty(obj_quad_next) ~= 1
        % ���obj_quad_next�ǿգ������reduced_MvTwSVM
        pi_temp = quadprog(obj_quad_next, obj_linear_next, A_next, B_next, [], [], lb, ub, [], opts);%�����ι滮����,���theta1
    else
        pi_temp = [];
    end
    
    % ��pi_next��δ֪��ָ�ֵ
    pi_next(idx_alpha_un, :) = pi_temp;
   % piȫ���������������һ���и��������и�ø��ֲ�
    pi_next = round(pi_next, num_truncation);
    
    % calculate the combined lagrangian multipler
    alpha_a = pi_next(idx_round_p, :);
    alpha_b = pi_next(l_p + idx_round_p, :);
    beta_p  = pi_next(2 * l_p + idx_round_n, :);
    beta_n  = pi_next(2 * l_p + l_n + idx_round_n, :);
    % ����Ѱ�Ҷ�Ӧ��֧��������
%     sv_index_a = find(alpha_a > 0 + num_accuracy );
%     sv_index_b = find(alpha_b > 0 + num_accuracy );
    
    % Ѱ������һ����Լ��������  
    index_beta_in_epsilon = find(beta_p + beta_n <= num_accuracy);  % ָ����|fA - fB| < epsilon�ı���
    index_beta_eq_epsilon = find(beta_p + beta_n > 0 + num_accuracy & beta_p + beta_n < D - num_accuracy);  % ָ����|fA - fB| = epsilon�ı���
    index_beta_out_epsilon = find(beta_p + beta_n >= D - num_accuracy);  % ָ����|fA - fB| > epsilon�ı���
    
    % �����ʽ��
    cesi_A = model.neg.cesi_A;
    cesi_B = model.neg.cesi_B;
    inv_A = model.neg.inv_A;
    inv_B = model.neg.inv_B;
    
    lag_piA_next = cesi_A * pi_next;
    lag_piB_next = cesi_B * pi_next;
    
    % calculate positive decision parameters
    uA = inv_A * lag_piA_next;
    uB = inv_B * lag_piB_next;
   
    % inherite the previous model
    model_next = model;
    model_next.neg.UB = C_next;
    
    % update pi 
    model_next.neg.pi = pi_next;
    model_next.neg.alpha_a = alpha_a;
    model_next.neg.alpha_b = alpha_b;
    model_next.neg.beta_p = beta_p;
    model_next.neg.beta_n = beta_n;
    
    model_next.vAn = uA;
    model_next.vBn = uB;
    
    model_next.Ca = Ca;
    model_next.Cb = Cb;
    model_next.D  = D;
    
    % support vector's index
%     model_next.neg.index_a = sv_index_a;
%     model_next.neg.index_b = sv_index_b;
    
    % epsilon vector's index
    model_next.neg.index_beta_in_epsilon  = index_beta_in_epsilon;
    model_next.neg.index_beta_eq_epsilon  = index_beta_eq_epsilon;
    model_next.neg.index_beta_out_epsilon = index_beta_out_epsilon;
    
    % ==== SSR record =====================================================
    SSR_indicator = ones(size(pi_next));
    SSR_indicator(idx_alpha_0, :) = 0;
    SSR_indicator(idx_alpha_C, :) = 1;
    SSR_indicator(idx_alpha_un, :) = 1/2;
    
    % screening alpha_A
    detailed.idx_alphaA_0 = idx_alphaA_0;
    detailed.idx_alphaA_C = idx_alphaA_C;
    detailed.idx_alphaA_un = idx_alphaA_un;
    
    % screening alpha_B
    detailed.idx_alphaB_0 = idx_alphaB_0;
    detailed.idx_alphaB_C = idx_alphaB_C;
    detailed.idx_alphaB_un = idx_alphaB_un;
    
    % screening beta +
    detailed.idx_betapn_0 = idx_betapn_0;
    detailed.idx_betap_D = idx_betap_D;
    detailed.idx_betap_un = idx_betap_un;
    
    % screening beta -
    detailed.idx_betanp_0 = idx_betanp_0;
    detailed.idx_betan_D = idx_betan_D;
    detailed.idx_betan_un = idx_betan_un;
    
    % total
    detailed.idx_alpha_0 = idx_alpha_0;
    detailed.idx_alpha_C = idx_alpha_C;
    detailed.idx_alpha_un = idx_alpha_un;
    
    % SSR rules
    detailed.SSR1 = SSR1;
    detailed.SSR2 = SSR2;
    detailed.SSR_indicator = SSR_indicator;
    detailed.screened_number = L - size(idx_alpha_un, 1);
    model_next.neg.SSR_detail = detailed;
    clearvars -except model_next detailed
end
