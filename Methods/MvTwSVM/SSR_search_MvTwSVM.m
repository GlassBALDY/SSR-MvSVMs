function [SSR_table, diagnosis] = SSR_search_MvTwSVM(trainXA, trainXB, trainy, testXA, testXB, testy, ...
    Box_Cons_p_series, Box_Cons_n_series, D_series, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon, varargin)
%   This function is used to implement Multi-view Twin SVM.
% Input:
%   XA: viewA's train data;
%   XB: viewB's train data;
%   y: train data's label;
%   Box_Cons_p/n: viewA/B's penalty parameter vector sequence with 2-by-L,
%   Box_Cons_p/n(1) = viewA's C_p/n, Box_Cons_p/n(2) = viewB's C_p/n
%   D: penalty parameter vector sequence for similarity constraint's slacks with 2-by-L.
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

% =========== main function =====================================
    global global_options

    % default value
    exe_date = datestr(now, 30);
    dir_ = ['SSR-MvTwSVM_result', '_', exe_date];
    result_name = 'SSR-MvTwSVM_table';
    Tikhonov = 1e-5;

    p = inputParser;
    addParameter(p,'directory', dir_);
    addParameter(p, 'result_name', result_name);
    addParameter(p, 'Tikhonov', Tikhonov);
    parse(p, varargin{:});

    % read parameter
    dir_ = p.Results.directory;   % a very tiny Tikhonov regulization term
    result_name = p.Results.result_name;
    Tikhonov = p.Results.Tikhonov;

    % create save direction
    mkdir(dir_)
    save_dir = [dir_, '\'];

    %% initialization
    % read the length of sequence
    L = size(D_series, 2);
    Acc = zeros(L, 2); Time = Acc; Screened_number = Acc;
    Acc_viewA = zeros(L, 2); Acc_viewB = zeros(L, 2);
    Screened_number_pos = Acc; Screened_number_neg = Acc;

    %% 1. calculate the first MvTwSVM
    Cp = Box_Cons_p_series(:, 1); Cn = Box_Cons_n_series(:, 1); D = D_series(:, 1);
    mvtwsvmmodel = MvTwsvmtrain(trainXA, trainXB, trainy, Cp, Cn, D, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, ...
        epsilon, 'Tikhonov', Tikhonov);
    [ ~, ~, ~, ~, ~, ~, ~, mvtwsvmmodel] = MvTwsvmpredict(mvtwsvmmodel, testXA, testXB, testy);

    % record Accuracy
    Acc(1, 1) = mvtwsvmmodel.accuracy;
    Acc(1, 2) = mvtwsvmmodel.accuracy;
    Acc_viewA(1, 1) = mvtwsvmmodel.accuracy_viewA;
    Acc_viewA(1, 2) = mvtwsvmmodel.accuracy_viewA;
    Acc_viewB(1, 1) = mvtwsvmmodel.accuracy_viewB;
    Acc_viewB(1, 2) = mvtwsvmmodel.accuracy_viewB;
    
    % record screened multipliers
    Screened_number(1, 1) = 0;
    Screened_number(1, 2) = 0;
    Screened_number_pos(1, 1) = 0;
    Screened_number_pos(1, 2) = 0;
    Screened_number_neg(1, 1) = 0;
    Screened_number_neg(1, 2) = 0;
    
    % record Time
    Time(1, 1) = mvtwsvmmodel.training_time;
    Time(1, 2) = mvtwsvmmodel.training_time;
    
    % record lagrangian multipliers
    mvtwsvmmodel.pos.Box_cons_series = Box_Cons_p_series;
    mvtwsvmmodel.pos.D_series = D_series(1, :);
    mvtwsvmmodel.pos.alpha_series = zeros(size(mvtwsvmmodel.pos.obj_linear, 1), size(mvtwsvmmodel.pos.D_series, 2));
    mvtwsvmmodel.pos.alpha_series(:, 1) = mvtwsvmmodel.pos.pi;
    
    mvtwsvmmodel.neg.Box_cons_series = Box_Cons_n_series;
    mvtwsvmmodel.neg.D_series = D_series(2, :);
    mvtwsvmmodel.neg.alpha_series = zeros(size(mvtwsvmmodel.neg.obj_linear, 1), size(mvtwsvmmodel.neg.D_series, 2));
    mvtwsvmmodel.neg.alpha_series(:, 1) = mvtwsvmmodel.neg.pi;
    
    % organize the table of results
    SSR_table = table(Box_Cons_p_series', Box_Cons_n_series', D_series', Acc, ...
        Acc_viewA, Acc_viewB, Time, Screened_number, Screened_number_pos, Screened_number_neg);
    SSR_table.Properties.VariableNames{1} = 'Box_Cons_p_series';
    SSR_table.Properties.VariableNames{2} = 'Box_Cons_n_series';
    SSR_table.Properties.VariableNames{3} = 'D_series';

    %% 2. substituting SSR process
    for i_step = 2: L
        Cp = Box_Cons_p_series(:, i_step); Cn = Box_Cons_n_series(:, i_step); D = D_series(:, i_step);
        %% SSR-MvTwSVM
        [mvtwsvmmodel_next, detailed] = DVI_rules_4_MvTwSVM(mvtwsvmmodel, [Cp', D(1, 1)], [Cn', D(2, 1)], global_options);
        %用predict函数计算模型准确度
        [ ~, ~, ~, ~, ~, ~, ~, mvtwsvmmodel_next] = MvTwsvmpredict(mvtwsvmmodel_next, testXA, testXB, testy);
        
        %% Raw-MvTwSVM
        raw_mvtwsvmmodel = MvTwsvmtrain(trainXA, trainXB, trainy, Cp, Cn, D, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon);
        %用predict函数计算模型准确度
        [ ~, ~, ~, ~, ~, ~, ~, raw_mvtwsvmmodel] = MvTwsvmpredict(raw_mvtwsvmmodel, testXA, testXB, testy);
        
        % record Accuracy
        Acc(i_step, 1) = mvtwsvmmodel_next.accuracy;
        Acc(i_step, 2) = raw_mvtwsvmmodel.accuracy;
        Acc_viewA(i_step, 1) = mvtwsvmmodel_next.accuracy_viewA;
        Acc_viewA(i_step, 2) = raw_mvtwsvmmodel.accuracy_viewA;
        Acc_viewB(i_step, 1) = mvtwsvmmodel_next.accuracy_viewB;
        Acc_viewB(i_step, 2) = raw_mvtwsvmmodel.accuracy_viewB;
        
        % record screened multipliers
        Screened_number(i_step, 1) = detailed.detailed_pos.screened_number + detailed.detailed_neg.screened_number;
        Screened_number(i_step, 2) = 0;
        Screened_number_pos(i_step, 1) = detailed.detailed_pos.screened_number;
        Screened_number_pos(i_step, 2) = 0;
        Screened_number_neg(i_step, 1) = detailed.detailed_neg.screened_number;
        Screened_number_neg(i_step, 2) = 0;
        
        % record Time
        Time(i_step, 1) = mvtwsvmmodel_next.training_time;
        Time(i_step, 2) = raw_mvtwsvmmodel.training_time;
        
        % record lagrangian multipliers
        mvtwsvmmodel_next.pos.alpha_series(:, i_step) = mvtwsvmmodel_next.pos.pi;
        mvtwsvmmodel_next.neg.alpha_series(:, i_step) = mvtwsvmmodel_next.neg.pi;
        
        %         diagnosis{i_step, 1} = mvtwsvmmodel_next; diagnosis{i_step, 2} = raw_mvtwsvmmodel;
        delta_alpha_pos(:, i_step * 3 - 2) = mvtwsvmmodel_next.pos.pi;
        delta_alpha_pos(:, i_step * 3 - 1) = raw_mvtwsvmmodel.pos.pi;
        delta_alpha_pos(:, i_step * 3) = mvtwsvmmodel_next.pos.SSR_detail.SSR_indicator;
        diagnosis.pos.delta_alpha = delta_alpha_pos;

        delta_alpha_neg(:, i_step * 3 - 2) = mvtwsvmmodel_next.neg.pi;
        delta_alpha_neg(:, i_step * 3 - 1) = raw_mvtwsvmmodel.neg.pi;
        delta_alpha_neg(:, i_step * 3) = mvtwsvmmodel_next.neg.SSR_detail.SSR_indicator;
        diagnosis.neg.delta_alpha = delta_alpha_neg;

        mvtwsvmmodel = mvtwsvmmodel_next;
        SSR_table = table(Box_Cons_p_series', Box_Cons_n_series', D_series', Acc, Acc_viewA, Acc_viewB, ...
            Time, Screened_number, Screened_number_pos, Screened_number_neg);
        SSR_table.Properties.VariableNames = {'Box_Cons_p_series', 'Box_Cons_n_series', ...
            'D_series', 'Accuracy', 'Accuracy_viewA', 'Accuracy_viewB', 'Time', ...
            'Screened_number', 'Screened_number_pos', 'Screened_number_neg'};
        save([save_dir, result_name, '.mat'], 'SSR_table');
        
        % diag
        if mod(i_step, 10) == 0
            
        end
    end
end

% Code diagnosis
% idx_bug = 2;
% SSR_model = diagnosis{idx_bug, 1};
% Raw_model = diagnosis{idx_bug, 2};