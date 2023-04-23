function [SSR_table, diagnosis] = SSR_search_SVM2K(trainXA, trainXB, trainy, testXA, testXB, testy, ...
    Box_Cons_series, D_series, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon, varargin)
%   This function is used to implement SSR-SVM-2K.
% Input:
%   trainXA: viewA's train data;
%   trainXB: viewB's train data;
%   trainy: train data's label;
%   testXA: viewA's test data;
%   testXB: viewB's test data;
%   testy: test data's label;
%   Box_Cons_series: viewA/B's penalty parameter vector sequence with
%   2-by-L array;
%   D_series: similarity constraint's penalty parameter vector sequence
%   with 1-by-L array;
%   kerneltypeA/B: viewA/B's kerneltype, kerneltypeA/B = 'linear', 'rbf',
%   'Gauss', 'poly';
%   kernelparamA/B: viewA/B's kernel parameter;
%   epsilon: similarity constraint;
% Alternative parameters:
%   'directory': the direcotry taht you want to store the SSR results;
%   'result_name': the name of the .mat file that contains the 'SSR_table';
% Output:
%   SSR_table: a table that contains the details of SSR results
% Usage:
%   [SSR_table] = SSR_search_SVM2K(trainXA, trainXB, trainy, testXA, testXB, testy, ...
%             Box_Cons_series, D_series, kernelA, kernelparamA, kernelB, kernelparamB, ...
%             epsilon, 'directory', 'SSR_SVM2K\', 'result_name', 'fold_1');

% =========== main function =====================================
    global global_options error
    exe_date = datestr(now, 30);
    dir_ = ['SSR-SVM2K_result', '_', exe_date];
    
    result_name = 'SSR-SVM2K_table';
    
    p = inputParser;
    addParameter(p, 'directory', dir_);
    addParameter(p, 'result_name', result_name);
    parse(p, varargin{:});
    
    % read parameter
    dir_ = p.Results.directory;   % a very tiny Tikhonov regulization term
    result_name = p.Results.result_name;
    
    % create save direction
    mkdir(dir_)
    save_dir = [dir_, '\'];
    
    %% initialization
    % read the length of sequence
    L = size(D_series, 2);
    Acc = zeros(L, 2); Time = Acc; Screened_number = Acc;
    Acc_viewA = zeros(L, 2); Acc_viewB = zeros(L, 2);
    
    %% 1. calculate the first SVM-2K
    C = Box_Cons_series(:, 1); D = D_series(:, 1);
    svm2kmodel = svm2ktrain(trainXA, trainXB, trainy, D, C, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon);
    [ ~, ~, ~, svm2kmodel] = svm2kpredict(svm2kmodel, testXA, testXB, testy);
    
    % record Accuracy
    Acc(1, 1) = svm2kmodel.accuracy;
    Acc(1, 2) = svm2kmodel.accuracy;
    Acc_viewA(1, 1) = svm2kmodel.accuracy_viewA;
    Acc_viewA(1, 2) = svm2kmodel.accuracy_viewA;
    Acc_viewB(1, 1) = svm2kmodel.accuracy_viewB;
    Acc_viewB(1, 2) = svm2kmodel.accuracy_viewB;
    
    % record screened multipliers
    Screened_number(1, 1) = 0;
    Screened_number(1, 2) = 0;
    
    % record Time
    Time(1, 1) = svm2kmodel.training_time;
    Time(1, 2) = svm2kmodel.training_time;
    SSR_table = table(Box_Cons_series', D_series', Acc, Acc_viewA, Acc_viewB, Time, Screened_number);
    SSR_table.Properties.VariableNames{1} = 'Box_Cons_series';
    SSR_table.Properties.VariableNames{2} = 'D_series';
    
    %% 2. substituting SSR process
    for i_step = 2: L
        C = Box_Cons_series(:, i_step); D = D_series(:, i_step);
        %% SSR-SVM-2K
        [svm2kmodel_next, detailed] = DVI_rules_4_SVM2K(svm2kmodel, [C', D(1, 1)], global_options, error);
        %用predict函数计算模型准确度
        [ ~, ~, ~, svm2kmodel_next] = svm2kpredict(svm2kmodel_next, testXA, testXB, testy);
        
        %% Raw-SVM-2K
        raw_svm2kmodel = svm2ktrain(trainXA, trainXB, trainy, D, C, kerneltypeA, kernelparamA, kerneltypeB, kernelparamB, epsilon);
        %用predict函数计算模型准确度
        [ ~, ~, ~, raw_svm2kmodel] = svm2kpredict(raw_svm2kmodel, testXA, testXB, testy);
        
        % record Accuracy
        Acc(i_step, 1) = svm2kmodel_next.accuracy;
        Acc(i_step, 2) = raw_svm2kmodel.accuracy;
        Acc_viewA(i_step, 1) = svm2kmodel_next.accuracy_viewA;
        Acc_viewA(i_step, 2) = raw_svm2kmodel.accuracy_viewA;
        Acc_viewB(i_step, 1) = svm2kmodel_next.accuracy_viewB;
        Acc_viewB(i_step, 2) = raw_svm2kmodel.accuracy_viewB;
        
        % record screened multipliers
        Screened_number(i_step, 1) = detailed.screened_number;
        Screened_number(i_step, 2) = 0;
        
        % record Time
        Time(i_step, 1) = svm2kmodel_next.training_time;
        Time(i_step, 2) = raw_svm2kmodel.training_time;
        
%         diagnosis{i_step, 1} = svm2kmodel_next; diagnosis{i_step, 2} = raw_svm2kmodel; 
        delta_alpha(:, i_step) = [svm2kmodel_next.alpha - raw_svm2kmodel.alpha];
        diagnosis.delta_alpha = delta_alpha;
        svm2kmodel = svm2kmodel_next;
        SSR_table = table(Box_Cons_series', D_series', Acc, Acc_viewA, Acc_viewB, ...
            Time, Screened_number);
        SSR_table.Properties.VariableNames = {'Box_Cons_series', ...
            'D_series', 'Accuracy', 'Accuracy_viewA', 'Accuracy_viewB', 'Time', 'Screened_number'};
        save([save_dir, result_name, '.mat'], 'SSR_table');
    end
end

% Code diagnosis
% idx_bug = 7;
% SSR_model = diagnosis{idx_bug, 1};
% Raw_model = diagnosis{idx_bug, 2};
% delta_alpha = [SSR_model.alpha - Raw_model.alpha]