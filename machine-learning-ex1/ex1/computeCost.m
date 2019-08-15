function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

m = length(y); % number of training examples
h = X * theta; % calculate hypothesis
diff = h - y; % take difference from actual "y" values
sum = diff' * diff; % take sum of squares
J = sum / (2 * m); % divide by 2m

end
