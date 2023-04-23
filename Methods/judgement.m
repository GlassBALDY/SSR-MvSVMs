function [accuracy, judgement_result] = judgement(predicted_label, real_label)
%	This function is used to obtain svmmodels accuracy.
% Input:
%   predicted_label: the label predicted by svmpredict function with n-by-1 vector;
%   real_label: test set's label or category with n-by-1 vector;
% Output:
%   accuracy: model's accuracy, calculated by (TP+TN)/(TP+FP+TN+FN);
%   judgement_result: detailed TP TN FP FN;

% - 18th August 2020;
% _ 17th August 2021, added precison, recall and F-1 score;

%======= main function ====================================================
    TP = 0; FP = 0; TN = 0; FN = 0;                    % preset judgement result
    f = predicted_label; testY = real_label;
    l = size(real_label, 1);
    for i = 1:l
        if f(i, 1) == 1 && testY(i, 1) == 1
            TP = TP + 1;
        elseif f(i, 1) == 1 && testY(i, 1) == -1
            FP = FP + 1;
        elseif f(i, 1) == -1 && testY(i, 1) == -1
            TN = TN + 1;
        else
            FN = FN + 1;
        end
    end
    accuracy = (TP + TN)/(TP + FP + TN + FN) * 100;          % calculate model'saccuracy
    recall = TP / (TP + FN) * 100;
    precision = TP / (TP + FP) * 100;
    F_1score = 2 * (precision * recall) / (precision + recall);
    G_means = sqrt((TP / (TP + FN)) * (TN / (TN + FP))) * 100;
    judgement_result.TP = TP;                          % store TP, TN, FP, FN
    judgement_result.TN = TN;
    judgement_result.FP = FP;
    judgement_result.FN = FN;
    judgement_result.accuracy = accuracy;
    judgement_result.recall = recall;
    judgement_result.precision = precision;
    judgement_result.F_1score = F_1score;
    judgement_result.G_means = G_means;
end

