function [ ] = myMvSSRplot(model, definition, plot_name)
%   This script is used to plot the decision boundary of two view svm model
%   ONLY IN TWO DIMENSION!
% Input:
%   model: the multi-view svm model that you want to plot;
%   definition: the definition of the graph, the smaller the value, the
%   higher the quality of graph, but cost more time.
% Output:
%   a graph which decision boundary on it.

% - Jiayi Zhu, 25th September 2020.
% - Jiayi Zhu, 7th October 2020. Adding MCPK module
% - Jiayi Zhu, 11th October 2020. Adding MvTHSVMs module
% - Jiayi Zhu, 14th October 2020. Fixing MCPK, MvTHSVM, SVM-2K
% - Jiayi Zhu, 24th October 2020. Fixing SVM+, SVM-2K
% - Jiayi Zhu, 25th june 2021, Adding CoupTHSVM
% - Jiayi Zhu, 28th September 2021, Adding SSR_SVM-2K and SSR_MvTwSVMs
% - Jiayi Zhu, 18th March 2022, Adding SSR_MvTHSVM
% =========================================================================
    % plot the graphic
    h = definition; % Mesh grid step size
    % obtain the name of the model
    name = model.name;
    % inspect the name of the model
    if nargin < 3
        plot_name = [];
    end
    %%
    if strcmp(name, 'SVM+')        % when name = 'SVM+'
        % import parameters from model
        XA = model.X;
        XB = model.X2;
        xp = model.xp;
        xn = model.xn;
        xp2 = model.xp2;
        xn2 = model.xn2;
        sv = XA(model.sv_index, :);
        psv = XB(model.psv_index, :);
        predict = model.predict;

        testX = model.testX;
        testxp = testX(model.testy == 1, :);
        testxn = testX(model.testy == -1, :);

        % mesh viewA and viewPriviledgeds' grid
        [Ax, Ay] = meshgrid(min(XA(:, 1)) : h: max(XA(:, 1)), min(XA(:, 2)): h: max(XA(:, 2)));
        [Bx, By] = meshgrid(min(XB(:, 1)) : h: max(XB(:, 1)), min(XB(:, 2)): h: max(XB(:, 2)));
        [~, svmplusscore, slackscore] = predict(model, [Ax(:), Ay(:)], [Bx(:), By(:)]);
        svmplusscoreGrid = reshape(svmplusscore(:, 1), size(Ax, 1), size(Ay, 2));
        slackscoreGrid = reshape(slackscore(:, 1), size(Bx, 1), size(By, 2));

        %% decision space
        % plot the decision space's graph
        figure               % create a layer
        title(model.name)       % print title
        hold on
        grid on

        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        % plot support vector
        plot(sv(:, 1), sv(:, 2), 'ko');

        % plot decision plane
        contour(Ax, Ay, svmplusscoreGrid, [1 1], 'color', 'r', 'Linestyle', '--');
        contour(Ax, Ay, svmplusscoreGrid, [-1 -1], 'color', 'b', 'Linestyle', '--');
        contour(Ax, Ay, svmplusscoreGrid, [0 0], 'color', 'k');

        if isempty(model.testy) == 0
            plot(testxp(:, 1), testxp(:, 2), 'r.');
            plot(testxn(:, 1), testxn(:, 2), 'b.');
        end

        legend('positive sample', 'negative sample', 'support vector', '+ decision plane', '- decision plane', 'decision plane', '+ test sample', '- test sample', 'Location', 'NorthEastOutside');
        hold off

        %% privilege space
        % plot the privilege space's correcting function
        figure               % create a layer
        title([model.name, ' privilege space'])       % print title
        hold on
        grid on
        % plot positive sample and negative sample
        plot(xp2(:, 1), xp2(:, 2), 'r+');
        plot(xn2(:, 1), xn2(:, 2), 'bx');
        % plot privilege support vector
        plot(psv(:, 1), psv(:, 2), 'ko');

        % plot decision plane
        contour(Bx, By, slackscoreGrid, [-0.1 0 0.1 0.3], 'ShowText', 'on');
        legend('positive sample', 'negative sample', 'support vector', 'slack variable plane', 'Location', 'NorthEastOutside');
        hold off

        %%

    elseif strcmp(name, 'SSR-SVM-2K') || strcmp(name, 'SVM-2K')   % when name = 'SVM-2K_SSR'
        % import parameters from model
        XA = model.CA;
        XB = model.CB;
        xp = model.CA(model.y == 1, :);
        xn = model.CA(model.y == -1, :);
        xp2 = model.CB(model.y == 1, :);
        xn2 = model.CB(model.y == -1, :);
        predict = model.predict;
        sva = model.support_vec_a;
        svb = model.support_vec_b;

        % import test set
        XAT = model.testXA;
        XBT = model.testXB;
        testy = model.testy;
        testXAp = XAT(testy == 1, :);
        testXAn = XAT(testy == -1, :);
        testXBp = XBT(testy == 1, :);
        testXBn = XBT(testy == -1, :);

        % import SSR detail， 导入SSR筛选过程
        % 导入索引，注意还需更正索引的起始
        idx_R_A = model.SSR_detail.idx_alphaA_0; idx_L_A = model.SSR_detail.idx_alphaA_C; idx_Eps_A = model.SSR_detail.idx_alphaA_un;
        idx_R_B = model.SSR_detail.idx_alphaB_0; idx_L_B = model.SSR_detail.idx_alphaB_C; idx_Eps_B = model.SSR_detail.idx_alphaB_un;
        idx_identified_A = [idx_R_A; idx_L_A]; idx_identified_B = [idx_R_B; idx_L_B];

        % 根据索引取值
        XA_SSR_R = XA(idx_R_A, :); XA_SSR_L = XA(idx_L_A, :);
        XB_SSR_R = XB(idx_R_B, :); XB_SSR_L = XB(idx_L_B, :);
        XA_identified = XA(idx_identified_A, :); XB_identified = XB(idx_identified_B, :);

        % mesh viewA and viewB's grid
        [Ax, Ay] = meshgrid(min(XA(:, 1)) : h: max(XA(:, 1)), min(XA(:, 2)): h: max(XA(:, 2)));
        [Bx, By] = meshgrid(min(XB(:, 1)) : h: max(XB(:, 1)), min(XB(:, 2)): h: max(XB(:, 2)));
        [~, svm2kscoreA, ~] = predict(model, [Ax(:), Ay(:)], []);
        [~, ~, svm2kscoreB] = predict(model, [], [Bx(:), By(:)]);
        svm2kscoreAGrid = reshape(svm2kscoreA(:, 1), size(Ax, 1), size(Ay, 2));
        svm2kscoreBGrid = reshape(svm2kscoreB(:, 1), size(Bx, 1), size(By, 2));

        %% ============== plot the viewA space's graph =====================
        %         figure               % create a layer
        subplot(1,2,1)
        title([model.name ' view A'])       % print title
        hold on
        grid on
        box on

        % 绘制被筛选出来的样本
        plot(XA_identified(:, 1), XA_identified(:, 2), 'ko', 'LineWidth', 1.2, 'markersize', 8);
        %         plot(XA_SSR_R(:, 1), XA_SSR_R(:, 2), 'ko', 'LineWidth', 1.2, 'markersize', 8);
        %         plot(XA_SSR_L(:, 1), XA_SSR_L(:, 2), 'go', 'LineWidth', 1.2, 'markersize', 8);

        % plot positive sample and negative sample, , 'HandleVisibility',
        % 'off'用于控制是否展示图例
        %         plot(xp(:, 1), xp(:, 2), 'r+', 'LineWidth', 1.2, 'markersize', 8, 'HandleVisibility', 'off');
        plot(xp(:, 1), xp(:, 2), 'r+', 'LineWidth', 1.2, 'markersize', 8);
        %         plot(xn(:, 1), xn(:, 2), 'bx', 'LineWidth', 1.2, 'markersize', 8, 'HandleVisibility', 'off');
        plot(xn(:, 1), xn(:, 2), 'bx', 'LineWidth', 1.2, 'markersize', 8);

        % plot support vector
        %         plot(sva(:, 1), sva(:, 2), 'ko', 'LineWidth', 1.2, 'markersize', 8);

        % plot viewA's decision plane
        contour(Ax, Ay, svm2kscoreAGrid, [1 1],   'color', 'r', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Ax, Ay, svm2kscoreAGrid, [-1 -1], 'color', 'b', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Ax, Ay, svm2kscoreAGrid, [0 0],   'color', 'k', 'LineWidth', 1.2);

        %         svm2kscoreAGrid_L = reshape(svm2kscoreA_L(:, 1), size(Ax, 1), size(Ay, 2));
        %         contourf(Ax, Ay, svm2kscoreAGrid, [-1, 1]);

        % if model.testy exist, draw test set
        %         if isempty(testy) == 0
        %             plot(testXAp(:, 1), testXAp(:, 2), 'r.');
        %             plot(testXAn(:, 1), testXAn(:, 2), 'b.');
        %         end
        legend('Screened Samples', '+ Training samples', '- Training Samples', 'Location', 'NorthEast');


        %         plot(XB_SSR_R(:, 1), XB_SSR_R(:, 2), 'mo');
        %         plot(XB_SSR_L(:, 1), XB_SSR_L(:, 2), 'mo');

        hold off

        %% ============== plot the viewB space's graph =====================
        %         figure               % create a layer
        subplot(1,2,2)
        title([model.name, ' view B'])       % print title
        hold on
        grid on
        box on

        % 绘制被筛选出来的样本
        plot(XB_identified(:, 1), XB_identified(:, 2), 'ko', 'LineWidth', 1.2, 'markersize', 8);
        %         plot(XB_SSR_R(:, 1), XB_SSR_R(:, 2), 'ko', 'LineWidth', 1.2, 'markersize', 8);
        %         plot(XB_SSR_L(:, 1), XB_SSR_L(:, 2), 'go', 'LineWidth', 1.2, 'markersize', 8);

        % plot positive sample and negative sample
        %         plot(xp2(:, 1), xp2(:, 2), 'r+', 'LineWidth', 1.2, 'markersize', 8, 'HandleVisibility', 'off');
        plot(xp2(:, 1), xp2(:, 2), 'r+', 'LineWidth', 1.2, 'markersize', 8);
        %         plot(xn2(:, 1), xn2(:, 2), 'bx', 'LineWidth', 1.2, 'markersize', 8, 'HandleVisibility', 'off');
        plot(xn2(:, 1), xn2(:, 2), 'bx', 'LineWidth', 1.2, 'markersize', 8);

        % plot support vector
        %         plot(svb(:, 1), svb(:, 2), 'ko', 'LineWidth', 1.2, 'markersize', 8);

        % plot decision plane
        contour(Bx, By, svm2kscoreBGrid, [1 1],   'color', 'r', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Bx, By, svm2kscoreBGrid, [-1 -1], 'color', 'b', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Bx, By, svm2kscoreBGrid, [0 0],   'color', 'k', 'LineWidth', 1.2);

        % if model.testy exist, draw test set
        %         if isempty(testy) == 0
        %             plot(testXBp(:, 1), testXBp(:, 2), 'r.');
        %             plot(testXBn(:, 1), testXBn(:, 2), 'b.');
        %         end
        legend('Screened Samples', '+ Training samples', '- Training Samples', 'Location', 'NorthEast');

        hold off
        %%

    elseif strcmp(name, 'Multi-view Twin SVM') || strcmp(name, 'MvTwSVM')	% when name = 'Multi-view Twin SVM'
        % import parameters from model
        % import train set
        XA = model.CA;
        XB = model.CB;
        % find positive/negative set
        xp = model.XAp;
        xn = model.XAn;
        xp2 = model.XBp;
        xn2 = model.XBn;

        % import support vector
        %             svap = model.svap; svbp = model.svbp;
        %             svan = model.svan; svbn = model.svbn;
        % import test set
        testXA = model.testXA;
        testXB = model.testXB;
        testy = model.testy;
        % find positive/negative set
        testXAp = testXA(model.testy == 1, :);
        testXAn = testXA(model.testy == -1, :);
        testXBp = testXB(model.testy == 1, :);
        testXBn = testXB(model.testy == -1, :);

        % import predict function
        predict = model.predict;

        %% calculate grid
        % mesh viewA and viewB's grid
        [Ax, Ay] = meshgrid(min(XA(:, 1)) -1 : h: max(XA(:, 1)) +1, min(XA(:, 2)) -1: h: max(XA(:, 2)) +1);
        [Bx, By] = meshgrid(min(XB(:, 1)) -1 : h: max(XB(:, 1)) +1, min(XB(:, 2)) -1: h: max(XB(:, 2)) +1);
        % calculate viewA/B's grid values
        [~, MvTwSVMscoreAp, MvTwSVMscoreAn, MvTwSVMscoreA, MvTwSVMscoreBp, MvTwSVMscoreBn, MvTwSVMscoreB, model] = predict(model, [Ax(:), Ay(:)], [Bx(:), By(:)]);
        % mesh viewA's grid
        MvTwSVMscoreApGrid = reshape(MvTwSVMscoreAp(:, 1), size(Ax, 1), size(Ay, 2));
        MvTwSVMscoreAnGrid = reshape(MvTwSVMscoreAn(:, 1), size(Ax, 1), size(Ay, 2));
        MvTwSVMscoreAGrid = reshape(MvTwSVMscoreA(:, 1), size(Ax, 1), size(Ay, 2));
        % mesh viewB's grid
        MvTwSVMscoreBpGrid = reshape(MvTwSVMscoreBp(:, 1), size(Bx, 1), size(By, 2));
        MvTwSVMscoreBnGrid = reshape(MvTwSVMscoreBn(:, 1), size(Bx, 1), size(By, 2));
        MvTwSVMscoreBGrid = reshape(MvTwSVMscoreB(:, 1), size(Bx, 1), size(By, 2));

        %%
        % plot the viewA's decision space's graph
        figure               % create a layer
        title([model.name ' view A'])       % print title
        hold on
        grid on

        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        % plot support vector
        %             plot(svap(:, 1), svap(:, 2), 'ko');
        %             plot(svan(:, 1), svan(:, 2), 'ko');

        % plot viewA's decision plane
        contour(Ax, Ay, MvTwSVMscoreApGrid, [-1 -1],   'color', 'b', 'Linestyle', '--');
        contour(Ax, Ay, MvTwSVMscoreAnGrid, [1 1],     'color', 'r', 'Linestyle', '--');
        contour(Ax, Ay, MvTwSVMscoreAGrid, [0 0],      'color', 'k');

        % if model.testy exist, draw test set
        if isempty(testy) == 0
            plot(testXAp(:, 1), testXAp(:, 2), 'r.');
            plot(testXAn(:, 1), testXAn(:, 2), 'b.');
        end

        legend('positive sample', 'negative sample', '+ decision plane', '- decision plane', 'decision plane', 'positive test sample', 'negative test sample', 'Location', 'NorthEastOutside');
        hold off

        %%
        % plot the viewB space's graph
        figure               % create a layer
        title([model.name, ' view B'])       % print title
        hold on
        grid on

        % plot positive sample and negative sample
        plot(xp2(:, 1), xp2(:, 2), 'r+');
        plot(xn2(:, 1), xn2(:, 2), 'bx');
        % plot privilege support vector
        %             plot(svbp(:, 1), svbp(:, 2), 'ko');
        %             plot(svbn(:, 1), svbn(:, 2), 'ko');

        % plot viewB's decision plane
        contour(Bx, By, MvTwSVMscoreBpGrid, [-1 -1],  'color', 'b', 'Linestyle', '--');
        contour(Bx, By, MvTwSVMscoreBnGrid, [1 1],    'color', 'r', 'Linestyle', '--');
        contour(Bx, By, MvTwSVMscoreBGrid, [0 0],     'color', 'k');

        % if model.testy exist, draw test set
        if isempty(testy) == 0
            plot(testXBp(:, 1), testXBp(:, 2), 'r.');
            plot(testXBn(:, 1), testXBn(:, 2), 'b.');
        end
        legend('positive sample', 'negative sample', '+ decision plane', '- decision plane', 'decision plane', 'positive test sample', 'negative test sample', 'Location','NorthEastOutside');
        hold off

    elseif strcmp(name, 'MvTwSVM\_SSR') || strcmp(name, 'MvTwSVM\_noSSR') || ...
            strcmp(name, 'SSR-MvTwSVM') % when name = 'Multi-view Twin SVM'
        % import parameters from model
        % import train set
        XA = model.CA;
        XB = model.CB;
        % find positive/negative set
        xp = model.XAp;
        xn = model.XAn;
        xp2 = model.XBp;
        xn2 = model.XBn;

        % import support vector
        %             svap = model.svap; svbp = model.svbp;
        %             svan = model.svan; svbn = model.svbn;
        % import test set
        testXA = model.testXA;
        testXB = model.testXB;
        testy = model.testy;
        % find positive/negative set
        testXAp = testXA(model.testy == 1, :);
        testXAn = testXA(model.testy == -1, :);
        testXBp = testXB(model.testy == 1, :);
        testXBn = testXB(model.testy == -1, :);

        % import predict function
        predict = model.predict;

        %% calculate grid
        % mesh viewA and viewB's grid
        [Ax, Ay] = meshgrid(min(XA(:, 1)) -1 : h: max(XA(:, 1)) +1, min(XA(:, 2)) -1: h: max(XA(:, 2)) +1);
        [Bx, By] = meshgrid(min(XB(:, 1)) -1 : h: max(XB(:, 1)) +1, min(XB(:, 2)) -1: h: max(XB(:, 2)) +1);
        % calculate viewA/B's grid values
        [~, MvTwSVMscoreAp, MvTwSVMscoreAn, MvTwSVMscoreA, MvTwSVMscoreBp, MvTwSVMscoreBn, MvTwSVMscoreB, model] = predict(model, [Ax(:), Ay(:)], [Bx(:), By(:)]);
        % mesh viewA's grid
        MvTwSVMscoreApGrid = reshape(MvTwSVMscoreAp(:, 1), size(Ax, 1), size(Ay, 2));
        MvTwSVMscoreAnGrid = reshape(MvTwSVMscoreAn(:, 1), size(Ax, 1), size(Ay, 2));
        MvTwSVMscoreAGrid = reshape(MvTwSVMscoreA(:, 1), size(Ax, 1), size(Ay, 2));
        % mesh viewB's grid
        MvTwSVMscoreBpGrid = reshape(MvTwSVMscoreBp(:, 1), size(Bx, 1), size(By, 2));
        MvTwSVMscoreBnGrid = reshape(MvTwSVMscoreBn(:, 1), size(Bx, 1), size(By, 2));
        MvTwSVMscoreBGrid = reshape(MvTwSVMscoreB(:, 1), size(Bx, 1), size(By, 2));

        %%
        % plot the viewA's decision space's graph
        subplot(1,2,1)
        title([model.name ' view A'])       % print title
        hold on
        box on
        grid on

        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        % plot support vector
        %             plot(svap(:, 1), svap(:, 2), 'ko');
        %             plot(svan(:, 1), svan(:, 2), 'ko');

        % plot viewA's decision plane
        contour(Ax, Ay, MvTwSVMscoreApGrid, [-1 -1],   'color', 'b', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Ax, Ay, MvTwSVMscoreAnGrid, [1 1],     'color', 'r', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Ax, Ay, MvTwSVMscoreAGrid,  [0 0],     'color', 'k', 'LineWidth', 1.2);

        % if model.testy exist, draw test set
        if isempty(testy) == 0
            plot(testXAp(:, 1), testXAp(:, 2), 'r.');
            plot(testXAn(:, 1), testXAn(:, 2), 'b.');
        end

        %         legend('positive sample', 'negative sample', '+ decision plane', '- decision plane', 'decision plane', 'positive test sample', 'negative test sample', 'Location', 'NorthEastOutside');
        hold off

        %%
        % plot the viewB space's graph
        subplot(1,2,2)
        title([model.name, ' view B'])       % print title
        hold on
        box on
        grid on

        % plot positive sample and negative sample
        plot(xp2(:, 1), xp2(:, 2), 'r+');
        plot(xn2(:, 1), xn2(:, 2), 'bx');
        % plot privilege support vector
        %             plot(svbp(:, 1), svbp(:, 2), 'ko');
        %             plot(svbn(:, 1), svbn(:, 2), 'ko');

        % plot viewB's decision plane
        contour(Bx, By, MvTwSVMscoreBpGrid, [-1 -1],  'color', 'b', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Bx, By, MvTwSVMscoreBnGrid, [1 1],    'color', 'r', 'Linestyle', '--', 'LineWidth', 1.4);
        contour(Bx, By, MvTwSVMscoreBGrid,  [0 0],    'color', 'k', 'LineWidth', 1.2);

        % if model.testy exist, draw test set
        if isempty(testy) == 0
            plot(testXBp(:, 1), testXBp(:, 2), 'r.');
            plot(testXBn(:, 1), testXBn(:, 2), 'b.');
        end
        %         legend('positive sample', 'negative sample', '+ decision plane', '- decision plane', 'decision plane', 'positive test sample', 'negative test sample', 'Location','NorthEastOutside');
        hold off
    elseif ( strcmp(name, 'MvTHSVM') || strcmp(name, 'MvTHSVM2C') ...
            || strcmp(name, 'ADMM-fastMvTHSVM') || strcmp(name, 'ADMM_fastMvTHSVM2C') ...
            || strcmp(name, 'scaled-MvTHSVM') || strcmp(name, 'SSR-MvTHSVM'))	% when name = 'MvTHSVM'
        % import parameters from model
        % import train set
        XA = model.CA;
        XB = model.CB;
        % find positive/negative set
        xp = model.CA(model.y == 1, :);
        xn = model.CA(model.y == -1, :);
        xp2 = model.CB(model.y == 1, :);
        xn2 = model.CB(model.y == -1, :);

        % import support vector
        svap = model.pos.svA; svbp = model.pos.svB;
        svan = model.neg.svA; svbn = model.neg.svB;
        % import test set
        testXA = model.testXA;
        testXB = model.testXB;
        testy = model.testy;
        % find positive/negative set
        testXAp = testXA(model.testy == 1, :);
        testXAn = testXA(model.testy == -1, :);
        testXBp = testXB(model.testy == 1, :);
        testXBn = testXB(model.testy == -1, :);

        % import predict function
        predict = model.predict;

        %% calculate grid
        % mesh viewA and viewB's grid
        [Ax, Ay] = meshgrid((min(XA(:, 1)) - min(XA(:, 1))/100) : h: (max(XA(:, 1)) + max(XA(:, 1))/100), ...
            (min(XA(:, 2)) - min(XA(:, 2))/100) : h: (max(XA(:, 2)) + max(XA(:, 2))/100));
        [Bx, By] = meshgrid((min(XB(:, 1)) - min(XB(:, 1))/100) : h: (max(XB(:, 1)) + max(XB(:, 1))/100), ...
            (min(XB(:, 2)) - min(XB(:, 2))/100) : h: (max(XB(:, 2)) + max(XB(:, 2))/100));
        % calculate viewA/B's grid values
        [~, MvTHSVMscoreAp, MvTHSVMscoreBp, MvTHSVMscoreAn, MvTHSVMscoreBn] = predict(model, [Ax(:), Ay(:)], [Bx(:), By(:)]);
        % mesh viewA's grid
        MvTHSVMscoreApGrid = reshape(MvTHSVMscoreAp(:, 1), size(Ax, 1), size(Ay, 2));
        MvTHSVMscoreAnGrid = reshape(MvTHSVMscoreAn(:, 1), size(Ax, 1), size(Ay, 2));
        % mesh viewB's grid
        MvTHSVMscoreBpGrid = reshape(MvTHSVMscoreBp(:, 1), size(Bx, 1), size(By, 2));
        MvTHSVMscoreBnGrid = reshape(MvTHSVMscoreBn(:, 1), size(Bx, 1), size(By, 2));

        %%
        % plot the viewA's decision space's graph
        supfigure = figure;               % create a layer
        hold on
        subplot(1,2,1)
        title([model.name ' view A'])       % print title
        hold on
        grid on
        box on
        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        % plot support vector
        plot(svap(:, 1), svap(:, 2), 'ko');
        plot(svan(:, 1), svan(:, 2), 'ko');

        % plot viewA's decision plane
        contour(Ax, Ay, MvTHSVMscoreApGrid, [1 1], 'color', 'm', 'LineWidth',1.5, 'Linestyle', '--');
        contour(Ax, Ay, MvTHSVMscoreAnGrid, [1 1], 'color', 'c', 'LineWidth',1.5, 'Linestyle', '--');
        contour(Ax, Ay, MvTHSVMscoreAnGrid - MvTHSVMscoreApGrid, [0 0], 'color', 'k', 'LineWidth',1.5);

        % if model.testy exist, draw test set
        if isempty(testy) == 0
            plot(testXAp(:, 1), testXAp(:, 2), 'r.');
            plot(testXAn(:, 1), testXAn(:, 2), 'b.');
        end

        %         legend('positive sample', 'negative sample', 'positive support vector', 'negative support vector', '+ decision plane', '- decision plane', 'decision plane', 'positive test sample', 'negative test sample', 'Location', 'NorthEastOutside');
        subfigure1 = gca;
        axis square;
        hold off

        %%
        % plot the viewB space's graph
        %         figure               % create a layer
        subplot(1,2,2)
        title([model.name, ' view B'])       % print title
        hold on
        grid on
        box on
        % plot positive sample and negative sample
        plot(xp2(:, 1), xp2(:, 2), 'r+');
        plot(xn2(:, 1), xn2(:, 2), 'bx');
        % plot privilege support vector
        plot(svbp(:, 1), svbp(:, 2), 'ko');
        plot(svbn(:, 1), svbn(:, 2), 'ko');

        % plot viewB's decision plane
        contour(Bx, By, MvTHSVMscoreBpGrid, [1 1], 'color', 'm', 'LineWidth',1.5, 'Linestyle', '--');
        contour(Bx, By, MvTHSVMscoreBnGrid, [1 1], 'color', 'c', 'LineWidth',1.5, 'Linestyle', '--');
        contour(Bx, By, MvTHSVMscoreBnGrid - MvTHSVMscoreBpGrid, [0 0], 'color', 'k', 'LineWidth',1.5);

        % if model.testy exist, draw test set
        if isempty(testy) == 0
            plot(testXBp(:, 1), testXBp(:, 2), 'r.');
            plot(testXBn(:, 1), testXBn(:, 2), 'b.');
        end
        %         legend('positive sample', 'negative sample', 'positive support vector', 'negative support vector', '+ decision plane', '- decision plane', 'decision plane', 'positive test sample', 'negative test sample', 'Location','NorthEastOutside');
        subfigure2 = gca;
        subfigure2.Position(1) = subfigure1.Position(1) + subfigure1.Position(3) + 0.04;
        subfigure2.Position(3) = subfigure1.Position(3);
        axis square;
        hold off
        hold off
        supfigure.Position(3) = supfigure.Position(3) * 2;
        fig = supfigure;
        % 这几条命令是控制图像打印时，尽量少打印白边的。
        % 我需要控制图中对象大小的命令
        % 实现了，通过gcf对象中的Position实现对图像中对象大小的调整
        fig.PaperPositionMode = 'auto';
        fig_pos = fig.PaperPosition;
        fig.PaperSize = [fig_pos(3) fig_pos(4)];

        if isempty(plot_name) == 1
            print([name, '_boundary'], '-dpdf', '-r1400')
            print([name, '_boundary'], '-depsc', '-r1400')
            print([name, '_boundary'], '-dmeta', '-r1400')
        elseif isempty(plot_name) ~= 1
            print([plot_name, '_boundary'], '-dpdf', '-r1400')
            print([plot_name, '_boundary'], '-depsc', '-r1400')
            print([plot_name, '_boundary'], '-dmeta', '-r1400')
        end
    end
end