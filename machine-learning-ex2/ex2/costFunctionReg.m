function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% ignore theta_0 for regularization
theta_0 = theta(2:length(theta));
J = -(1/m)*(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)))+(lambda/(2*m))*(theta_0'*theta_0);
grad = (1/m)*(X'*(sigmoid(X*theta)-y))+(lambda/m)*theta;

% ignore theta_0 for regularization - no regularization term
grad(1) = (1/m)*(X(:,1)'*(sigmoid(X*theta)-y));
end
