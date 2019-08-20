function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

theta_0 = theta(2:length(theta)); % ignore theta 0 in regularization
J = (-1/m)*(y'*log(sigmoid(X*theta))+(1-y)'*log(1-sigmoid(X*theta)))+lambda/(2*m)*(theta_0'*theta_0);
grad = (1/m)*(X'*(sigmoid(X*theta)-y))+(lambda/m)*theta;
grad(1) = (1/m)*(X(:,1)'*(sigmoid(X*theta)-y)); % theta 0 has no regularization term

end
