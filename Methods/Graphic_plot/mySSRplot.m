function [ ] = mySSRplot(model, definition)
%   This script is used to plot the decision boundary of one view svm model
%   ONLY IN TWO DIMENSION! 
% Input:
%   model: the svm model that you want to plot;
%   definition: the definition of the graph, the smaller the value, the
%               higher the quality of graph, but cost more time.
% Output:
%   a graph which decision boundary on it.

% - Update by Jiayi Zhu, 13th October 2020 (Containing 'LSSVM', 'SVM', 'Twin SVM')
% - Update by Jiayi Zhu, 14th October 2020 (adding 'THSVM', fixing 'Twin SVM')
% - Update by Jiayi Zhu, 25th October 2020 (adding 'Location', 'NortheastOutside')
% - Update by Jiayi Zhu, 29th October 2020 (adding 'one-class v- SVM', 'Single Sphere')
% - Update by Jiayi Zhu, 9th November 2021 (reforming 'THSVM')

% ========= main function =================================================
    name = model.name;
    % detect the model's name
    if strcmp(name, 'LSSVM') == 1       % when name = 'LSSVM', draw lssvm's graph          
        % plot the graphic
        h = definition; % Mesh grid step size
        % Import data from model
        X = model.X;    % train samples
        xp = model.xp;  % +1 train patterns
        xn = model.xn;  % -1 train patterns
        
        XT = model.testX;   % test samples
        yT = model.testy;   % test category
        xtp = XT(yT == 1, :);   % +1 test patterns
        xtn = XT(yT == -1, :);  % -1 test patterns
        
        predict = model.predict;    % prediction function
        sv_index = model.sv_index;  % support vector's index
        sv = X(sv_index, :);    % reload support vector
        [X1, X2] = meshgrid(min(X(:, 1)) : h: max(X(:, 1)), min(X(:, 2)): h: max(X(:, 2)));
        [~, svmscore] = predict(model, [X1(:), X2(:)]);
        svmscoreGrid = reshape(svmscore(:, 1), size(X1, 1), size(X2, 2));

        % plot the graph
        figure              % create a layer
        title(model.name)   % print title
        hold on
        grid on
        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        % plot support vector
        % plot(sv(:, 1), sv(:, 2), 'ko');
        % plot decision plane
        contour(X1, X2, svmscoreGrid, [1 1], 'ShowText', 'on', 'color', 'b', 'Linestyle', '--');
        contour(X1, X2, svmscoreGrid, [-1 -1], 'ShowText', 'on', 'color', 'r', 'Linestyle', '--');
        contour(X1, X2, svmscoreGrid, [0 0], 'ShowText', 'on', 'color', 'k');
        
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
        legend('positive sample', 'negative sample', '+1 decision plane', '-1 decision plane', 'decision plane', 'positive test', 'negative test', 'Location', 'NorthEastOutside');
        hold off
    %%    
    elseif strcmp(name, 'SSR-SVM') == 1     % when name = 'SVM', draw svm's graph 
        % plot the graphic
        h = definition; % Mesh grid step size
        % Import data from model
        X = model.X;    % train samples
        xp = model.xp;  % +1 train patterns
        xn = model.xn;  % -1 train patterns
        
        XT = model.testX;   % test samples
        yT = model.testy;   % test category
        xtp = XT(yT == 1, :);   % +1 test patterns
        xtn = XT(yT == -1, :);  % -1 test patterns
        
        predict = model.predict;    % prediction function
        sv_index = model.sv_index;  % support vector's index
        sv = X(sv_index, :);    % reload support vector
        [X1, X2] = meshgrid(min(X(:, 1)) : h: max(X(:, 1)), min(X(:, 2)): h: max(X(:, 2)));
        [~, svmscore] = predict(model, [X1(:), X2(:)]);
        svmscoreGrid = reshape(svmscore(:, 1), size(X1, 1), size(X2, 2));

        % plot the graph
        figure              % create a layer
        title(model.name)   % print title
        hold on
        grid on

        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');

        % plot support vector
        plot(sv(:, 1), sv(:, 2), 'ko');

        % plot decision plane
        contour(X1, X2, svmscoreGrid, [1 1], 'color', 'r', 'Linestyle', '--', 'LineWidth', 1.2);
        contour(X1, X2, svmscoreGrid, [-1 -1], 'color', 'b', 'Linestyle', '--', 'LineWidth', 1.2);
        contour(X1, X2, svmscoreGrid, [0 0], 'color', 'k', 'LineWidth', 1.2);

        % plot positive test and negative test
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
        legend('positive sample', 'negative sample', 'support vector', ...
            'screened out alpha_{i} = 0', 'screened out alpha_{i} = c', ...
            '+1 decision plane', '-1 decision plane', 'decision plane', ...
            'positive test', 'negative test', 'Location', 'NortheastOutside');
        hold off
    %%    
    elseif strcmp(name, 'SSR-TwSVM') == 1    % when name = 'Twin SVM', draw svm's graph 
        % plot the graphic
        h = definition; % Mesh grid step size
        
        C = model.C;    % train samples
        xp = model.xp;  % +1 train patterns
        xn = model.xn;  % -1 train patterns
        
        XT = model.testX;   % test samples
        yT = model.testy;   % test category
        xtp = XT(yT == 1, :);   % +1 test patterns
        xtn = XT(yT == -1, :);  % -1 test patterns
        % caution!! positive support vectors are indicated by negative samples
        % and vice versa
        svxp = xp(model.svindex_n, :);
        svxn = xn(model.svindex_p, :);
        
        % screened alpha_0
        ssr_0_p = xp(model.neg.SSR_detail.idx_alpha_0, :);
        ssr_0_n = xn(model.pos.SSR_detail.idx_alpha_0, :);
        ssr_0 = [ssr_0_p; ssr_0_n];
        
        % screened alpha c
        ssr_c_p = xp(model.neg.SSR_detail.idx_alpha_c, :);
        ssr_c_n = xn(model.pos.SSR_detail.idx_alpha_c, :);
        ssr_c = [ssr_c_p; ssr_c_n];
        
        %
        predict = model.predict;
        [X1, X2] = meshgrid(min(C(:, 1)) : h: max(C(:, 1)), min(C(:, 2)): h: max(C(:, 2)));
        [~, twsvmscore, twpscore, twnscore] = predict(model, [X1(:), X2(:)]);
        twsvmscoreGrid = reshape(twsvmscore(:, 1), size(X1, 1), size(X2, 2));
        twsvmpscoreGrid = reshape(twpscore(:, 1), size(X1, 1), size(X2, 2));
        twsvmnscoreGrid = reshape(twnscore(:, 1), size(X1, 1), size(X2, 2));

        figure
        title(model.name)
        hold on
        grid on
        
        % draw positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        
        % draw support vector
        plot(svxp(:, 1), svxp(:, 2), 'ro', svxn(:, 1), svxn(:, 2),'bo');
        
        % draw ssr 0
        plot(ssr_0(:, 1), ssr_0(:, 2), 'ms', 'MarkerSize', 12);
        
        % draw ssr c
        plot(ssr_c(:, 1), ssr_c(:, 2), 'c^', 'MarkerSize', 12);
        
        % draw decision plane
        contour(X1, X2, twsvmpscoreGrid, [-1 -1], 'color', 'r', 'LineWidth', 1.0);
        contour(X1, X2, twsvmnscoreGrid, [1 1], 'color', 'b', 'LineWidth', 1.0);
        contour(X1, X2, twsvmscoreGrid, [0 0], 'color', 'k', 'LineWidth', 1.4);
        
        % plot positive test and negative test
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
        legend('positive sample', 'negative sample', '+1 support vector', '-1 support vector', ...
            'screened out \alpha_{i} = 0', 'screened out \alpha_{i} = c', ...
            '+1 decision plane', '-1 decision plane', 'decision plane', 'positive test', 'negative test', 'Location', 'NorthEastOutside');
        hold off
    %%    
    elseif strcmp(name, 'SSR-THSVM')
        h = definition; % Mesh grid step size
        
        % import train set
        C = model.C;
        xp = model.xp;
        xn = model.xn;
        
        % import test set
        XT = model.testX;
        yT = model.testy;
        xtp = XT(yT == 1, :);
        xtn = XT(yT == -1, :);
        % import support vector
        svxp = xp(model.pos.svindex, :);
        svxn = xn(model.neg.svindex, :);
        predict = model.predict;
        [X1, X2] = meshgrid(min(C(:, 1)) : h: max(C(:, 1)), min(C(:, 2)): h: max(C(:, 2)));
        [~, thsvmscore, thpscore, thnscore] = predict(model, [X1(:), X2(:)]);
        twsvmscoreGrid = reshape(thsvmscore(:, 1), size(X1, 1), size(X2, 2));
        twsvmpscoreGrid = reshape(thpscore(:, 1), size(X1, 1), size(X2, 2));
        twsvmnscoreGrid = reshape(thnscore(:, 1), size(X1, 1), size(X2, 2));

        figure
        title(model.name)
        hold on
        grid on
        box on
        % draw positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');
        % draw support vector
        plot(svxp(:, 1), svxp(:, 2), 'ro', svxn(:, 1), svxn(:, 2),'bo');
        % draw decision plane
        contour(X1, X2, twsvmpscoreGrid, [1 1], 'color', 'r', 'Linestyle', '--', 'LineWidth', 1.2);
        contour(X1, X2, twsvmnscoreGrid, [1 1], 'color', 'b', 'Linestyle', '--', 'LineWidth', 1.2);
        contour(X1, X2, twsvmscoreGrid, [0 0], 'color', 'k', 'LineWidth', 1.4);
         % plot positive test and negative test
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
%         legend('positive sample', 'negative sample', '+1 support vector', '-1 support vector', '+1 hyper-sphere', '-1 hyper-sphere', 'decision curve', 'positive test', 'negative test', 'Location', 'NorthEastOutside');
        axis square;
        hold off
		    %%    
    elseif strcmp(name, 'One-Class v-SVM') == 1
        % plot the graphic
        h = definition; % Mesh grid step size
        % Import data from model
        X = model.X;    % train samples
        rho = model.rho;
        
        XT = model.testX;   % test samples
        yT = model.testy;   % test category
        xtp = XT(yT == 1, :);   % +1 test patterns
        xtn = XT(yT == -1, :);  % -1 test patterns
        
        predict = model.predict;    % prediction function
        sv = model.sv;    % reload support vector
        [X1, X2] = meshgrid(min(X(:, 1)) -1: h: max(X(:, 1)) +1, min(X(:, 2)) -1: h: max(X(:, 2)) +1);
        [~, ocvsvmscore] = predict(model, [X1(:), X2(:)]);
        ocvsvmscoreGrid = reshape(ocvsvmscore(:, 1), size(X1, 1), size(X2, 2));

        % plot the graph
        figure              % create a layer
        title(model.name)   % print title
        hold on
        grid on
        % plot positive sample and negative sample
        plot(X(:, 1), X(:, 2), 'r+');
        % plot support vector
        plot(sv(:, 1), sv(:, 2), 'ko');
        % plot decision plane
        contour(X1, X2, ocvsvmscoreGrid, [0 0], 'ShowText', 'on', 'color', 'k');
        
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
        legend('train sample', 'support vector', 'rho-decision plane', 'positive test', 'negative test');
        hold off
    %%
    elseif strcmp(name, 'Single Sphere') == 1
        % plot the graphic
        h = definition; % Mesh grid step size
        % Import data from model
        X = model.X;    % train samples
        d = model.d;    % margin dist
        
        XT = model.testX;   % test samples
        yT = model.testy;   % test category
        xtp = XT(yT == 1, :);   % +1 test patterns
        xtn = XT(yT == -1, :);  % -1 test patterns
        
        predict = model.predict;    % prediction function
        sv = model.sv;    % reload support vector
        [X1, X2] = meshgrid(min(X(:, 1)) -1: h: max(X(:, 1)) +1, min(X(:, 2)) -1: h: max(X(:, 2)) +1);
        [~, SSsvmscore] = predict(model, [X1(:), X2(:)]);
        SSsvmscoreGrid = reshape(SSsvmscore(:, 1), size(X1, 1), size(X2, 2));

        % plot the graph
        figure              % create a layer
        title(model.name)   % print title
        hold on
        grid on
        % plot positive sample and negative sample
        plot(X(:, 1), X(:, 2), 'r+');
        % plot support vector
        plot(sv(:, 1), sv(:, 2), 'ko');
        % plot decision plane
        contour(X1, X2, SSsvmscoreGrid, [d d], 'ShowText', 'on', 'color', 'r', 'Linestyle', '--');
        contour(X1, X2, SSsvmscoreGrid, [-d -d], 'ShowText', 'on', 'color', 'b', 'Linestyle', '--');
        contour(X1, X2, SSsvmscoreGrid, [0 0], 'ShowText', 'on', 'color', 'k');
        
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
        legend('train sample', 'support vector', '+ decision plane', '- decision plane', 'decision plane','positive test', 'negative test');
        hold off 
    
    elseif strcmp(name, 'Pin-SVM') == 1
        % plot the graphic
        h = definition; % Mesh grid step size
        % Import data from model
        X = model.trainX;    % train samples
        xp = X(model.trainy == 1, :);  % +1 train patterns
        xn = X(model.trainy == -1, :);  % -1 train patterns
        
        XT = model.testX;   % test samples
        yT = model.testy;   % test category
        xtp = XT(yT == 1, :);   % +1 test patterns
        xtn = XT(yT == -1, :);  % -1 test patterns
        
        predict = model.predict;    % prediction function
        sv_index = model.sv_index;  % support vector's index
        sv = X(sv_index, :);    % reload support vector
        [X1, X2] = meshgrid(min(X(:, 1)) : h: max(X(:, 1)), min(X(:, 2)): h: max(X(:, 2)));
        [~, pinsvmscore] = predict(model, [X1(:), X2(:)]);
        pinsvmscoreGrid = reshape(pinsvmscore(:, 1), size(X1, 1), size(X2, 2));

        % plot the graph
        figure              % create a layer
        title(model.name)   % print title
        hold on
        grid on

        % plot positive sample and negative sample
        plot(xp(:, 1), xp(:, 2), 'r+');
        plot(xn(:, 1), xn(:, 2), 'bx');

        % plot support vector
        plot(sv(:, 1), sv(:, 2), 'ko');

        % plot decision plane
        contour(X1, X2, pinsvmscoreGrid, [1 1], 'ShowText', 'on', 'color', 'r', 'Linestyle', '--');
        contour(X1, X2, pinsvmscoreGrid, [-1 -1], 'ShowText', 'on', 'color', 'b', 'Linestyle', '--');
        contour(X1, X2, pinsvmscoreGrid, [0 0], 'ShowText', 'on', 'color', 'k');

        % plot positive test and negative test
        if isempty(yT) == 0     % if model.testy isn't null, then draw test set.
            plot(xtp(:, 1), xtp(:, 2), 'r.');
            plot(xtn(:, 1), xtn(:, 2), 'b.');
        end
        legend('positive sample', 'negative sample', 'support vector', '+1 decision plane', '-1 decision plane', 'decision plane', 'positive test', 'negative test', 'Location', 'NortheastOutside');
        hold off        
    end
    box on;
    grid on;
    fig = gcf;
%     fig.PaperPositionMode = 'auto';
    % 一条从知乎上抄的代码
    % 原地址：https://zhuanlan.zhihu.com/p/57606534
    set(gca, 'Position', get(gca, 'OuterPosition') - get(gca, 'TightInset') * [-1 0 2.7 0; 0 -1 0 1.5; 0 0 1 0; 0 0 0 1]);
    fig_pos = fig.PaperPosition;
    fig.PaperSize = [fig_pos(3) fig_pos(4)];
    print('-painters', '-dpdf', '-r1400') % 打印pdf图
end

