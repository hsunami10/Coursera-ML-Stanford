function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

m = size(X, 1);
n = size(X, 2);
all_theta = zeros(num_labels, n + 1); % n+1 - theta_0

% Add ones to the X data matrix - first column X_0
X = [ones(m, 1) X];

initial_theta = zeros(n+1, 1); % Initialize fitting parameters
options = optimset('GradObj', 'on', 'MaxIter', 50); % Set Options

% find thetas / parameters associated with each class 1:10
% logical vector y == i has to only have values 0 OR 1
for i=1:num_labels
    [theta] = fmincg(@(t)(lrCostFunction(t, X, y == i, lambda)), initial_theta, options); % find optimal theta with optimized algorithm
    all_theta(i,:) = theta; % save into rows of theta matrix
end

end
