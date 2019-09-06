function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
H = X * theta;
J = 1/(2*m)*((H-y)'*(H-y))+(lambda/(2*m))*theta(2:end)'*theta(2:end);

grad = 1/m*(X'*(H-y))+(lambda/m)*theta;
grad(1) = 1/m*(X(:,1)'*(H-y));

end
