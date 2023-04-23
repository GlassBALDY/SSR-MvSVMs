clear all
addpath('..\Methods\', '..\Methods\MvTwSVM\');

%% SSR-MvTwSVM for AwA data sets
% experiment setting:
% kernel: linear
% parameter interval: 0.1: 0.1: 10
% num_accuracy = 1e-5;
% num_truncation = 4;
% cAp = cBp = cAn = cBn = Dp = Dn;
% epsilon = 0.1;
% Tikhonov = 1e-1;

%% Hyper-parameter
% generate parameter sequence
i  = 2 .^ [-1: 0.1: 1];
CA = i;
CB = i;
D  = i;

Box_Cons_p_series = [CA; CB];
Box_Cons_n_series = [CA; CB];
D_series = [D; D];

% The rest of the parameters
epsilon = 0.1;
kernelA = 'Gauss';
kernelparamA = 4;
kernelB = 'Gauss';
kernelparamB = 4;

%% Data pre-processing
normalization_method = '01';
k_fold_num = 5;

%% Load Toy data set
data_dir_name = '..\Data\synthetic_multiview_data\';
data_dir_structure = dir(data_dir_name);

num_task = size(data_dir_structure, 1) - 2;

for idx_task = 1: num_task
    data_set_name = data_dir_structure(idx_task + 2).name;
    %% benchmark beginning, load data set
    load([data_dir_name, data_set_name])
    name = task_1.name;
    XA = task_1.XA; XB = task_1.XB; y = task_1.y;
    XA = data_normalization(XA, normalization_method); XB = data_normalization(XB, normalization_method);
    
    %% DO k-FOLD CROSS VALIDATION
    kindices = k_foldgenerator(k_fold_num, size(XA, 1));     % cross validation indices
    
    for idx_fold = 1: k_fold_num
        test_index = (kindices == idx_fold);
        train_index = ~test_index;
        trainXA = XA(train_index, :); testXA = XA(test_index, :);
        trainXB = XB(train_index, :); testXB = XB(test_index, :);
        trainy = y(train_index, :); testy = y(test_index, :);
        
        [SSR_table] = SSR_search_MvTwSVM(trainXA, trainXB, trainy, testXA, testXB, testy, ...
            Box_Cons_p_series, Box_Cons_n_series, D_series, ...
            kernelA, kernelparamA, kernelB, kernelparamB, epsilon, ...
            'directory', ['SSR_MvTwSVM_Toy\Toy_', name, '\'], 'result_name', ['fold_', num2str(idx_fold)]);
    end
end

%% extract the SSR_tables
%% SSR-MvTwSVM's result extractor
% set the directory of results
data_dir = dir('.\SSR_MvTwSVM_Toy\');
data_name = {data_dir.name};
data_name = data_name(3: end)';
results_dir = '.\SSR_MvTwSVM_Toy\';

% set the length of parameter sequence
% parameter_length = 100;
% set the number of folds
k_fold = 5;

for idx_data = 1 : size(data_name, 1)
    task_name = sprintf('%s%s%s', results_dir, data_name{idx_data, 1}, '\');
    task_dir = dir(task_name);
    fold_name = {task_dir.name};
    fold_name = fold_name(3: end)';
    
    for idx_fold = 1: k_fold
        if idx_fold == 1
            fold_temp  = sprintf('%s%s', task_name, fold_name{idx_fold});
            load(fold_temp);
            parameter_length = size(SSR_table, 1);
            SSR_Accu_fold = zeros(parameter_length, k_fold);
            SSR_Accu_viewA_fold = zeros(parameter_length, k_fold);
            SSR_Accu_viewB_fold = zeros(parameter_length, k_fold);
            Raw_Accu_fold = zeros(parameter_length, k_fold);
            Raw_Accu_viewA_fold = zeros(parameter_length, k_fold);
            Raw_Accu_viewB_fold = zeros(parameter_length, k_fold);
            SSR_Time_fold = zeros(parameter_length, k_fold);
            Raw_Time_fold = zeros(parameter_length, k_fold);
            SSR_screened_fold = zeros(parameter_length, k_fold);
            SSR_screened_pos_fold = zeros(parameter_length, k_fold);
            SSR_screened_neg_fold = zeros(parameter_length, k_fold);

        end
        fold_temp  = sprintf('%s%s', task_name, fold_name{idx_fold});
        load(fold_temp);
        SSR_Accu_fold(:, idx_fold) = SSR_table.Accuracy(:, 1);
        SSR_Accu_viewA_fold(:, idx_fold) = SSR_table.Accuracy_viewA(:, 1);
        SSR_Accu_viewB_fold(:, idx_fold) = SSR_table.Accuracy_viewB(:, 1);
        
        Raw_Accu_fold(:, idx_fold) = SSR_table.Accuracy(:, 2);
        Raw_Accu_viewA_fold(:, idx_fold) = SSR_table.Accuracy_viewA(:, 2);
        Raw_Accu_viewB_fold(:, idx_fold) = SSR_table.Accuracy_viewB(:, 2);
        
        SSR_Time_fold(:, idx_fold) = SSR_table.Time(:, 1);
        Raw_Time_fold(:, idx_fold) = SSR_table.Time(:, 2);
        SSR_screened_fold(:, idx_fold) = SSR_table.Screened_number(:, 1);
        try
            SSR_screened_pos_fold(:, idx_fold) = SSR_table.Screened_number_pos(:, 1);
            SSR_screened_neg_fold(:, idx_fold) = SSR_table.Screened_number_neg(:, 1);
        catch
            SSR_screened_fold(:, idx_fold) = SSR_table.Screened_number(:, 1);
        end
    end
    
    % calculate the k-fold's mean and std
    SSR_mean_Accu = mean(SSR_Accu_fold, 2);
    SSR_std_Accu  = std(SSR_Accu_fold, 0, 2);
    SSR_mean_Accu_viewA = mean(SSR_Accu_viewA_fold, 2);
    SSR_std_Accu_viewA  = std(SSR_Accu_viewA_fold, 0, 2);
    SSR_mean_Accu_viewB = mean(SSR_Accu_viewB_fold, 2);
    SSR_std_Accu_viewB  = std(SSR_Accu_viewB_fold, 0, 2);
    
    SSR_mean_Time = mean(SSR_Time_fold, 2);
    SSR_std_Time  = std(SSR_Time_fold, 0, 2);
    
    Raw_mean_Accu = mean(Raw_Accu_fold, 2); 
    Raw_std_Accu  = std(Raw_Accu_fold, 0, 2);
    Raw_mean_Accu_viewA = mean(Raw_Accu_viewA_fold, 2); 
    Raw_std_Accu_viewA  = std(Raw_Accu_viewA_fold, 0, 2);
    Raw_mean_Accu_viewB = mean(Raw_Accu_viewB_fold, 2); 
    Raw_std_Accu_viewB  = std(Raw_Accu_viewB_fold, 0, 2);
    
    Raw_mean_Time = mean(Raw_Time_fold, 2);
    Raw_std_Time  = std(Raw_Time_fold, 0, 2);
    
    SSR_mean_screened = mean(SSR_screened_fold, 2);
    try
        SSR_mean_screened_pos = mean(SSR_screened_pos_fold, 2);
        SSR_mean_screened_neg = mean(SSR_screened_neg_fold, 2);
    catch
        SSR_mean_screened = mean(SSR_screened_fold, 2);
    end
    
    % store the k-fold results
    try
        Box_Cons_p_series = SSR_table.Box_Cons_p_series;
        Box_Cons_n_series = SSR_table.Box_Cons_n_series;
        D_series = SSR_table.D_series;
        SSR_mean_table = table(Box_Cons_p_series, Box_Cons_n_series, D_series, ...
            Raw_mean_Accu, Raw_std_Accu, ...
            Raw_mean_Accu_viewA, Raw_std_Accu_viewA, ...
            Raw_mean_Accu_viewB, Raw_std_Accu_viewB, Raw_mean_Time, ...
            SSR_mean_Accu, SSR_std_Accu, ...
            SSR_mean_Accu_viewA, SSR_std_Accu_viewA, ...
            SSR_mean_Accu_viewB, SSR_std_Accu_viewB, SSR_mean_Time, ...
            SSR_mean_screened, SSR_mean_screened_pos, SSR_mean_screened_neg);
    catch
        Box_Cons_series = SSR_table.Box_Cons_series;
        D_series = SSR_table.D_series;
        SSR_mean_table = table(Box_Cons_series, D_series, ...
            Raw_mean_Accu, Raw_std_Accu, ...
            Raw_mean_Accu_viewA, Raw_std_Accu_viewA, ...
            Raw_mean_Accu_viewB, Raw_std_Accu_viewB, Raw_mean_Time, ...
            SSR_mean_Accu, SSR_std_Accu, ...
            SSR_mean_Accu_viewA, SSR_std_Accu_viewA, ...
            SSR_mean_Accu_viewB, SSR_std_Accu_viewB, SSR_mean_Time, ...
            SSR_mean_screened);
    end
    save([task_name, 'fold_mean.mat'], 'SSR_mean_table');
end

%%
rmpath('..\Methods\', '..\Methods\MvTwSVM\');