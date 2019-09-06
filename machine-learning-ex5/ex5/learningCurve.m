function [error_train, error_val] = ...
    learningCurve(X, y, Xval, yval, lambda)
%LEARNINGCURVE Generates the train and cross validation set errors needed 
%to plot a learning curve
%   [error_train, error_val] = ...
%       LEARNINGCURVE(X, y, Xval, yval, lambda) returns the train and
%       cross validation set errors for a learning curve. In particular, 
%       it returns two vectors of the same length - error_train and 
%       error_val. Then, error_train(i) contains the training error for
%       i examples (and similarly for error_val(i)).
%
%   In this function, you will compute the train and test errors for
%   dataset sizes from 1 up to m. In practice, when working with larger
%   datasets, you might want to do this in larger intervals.
%

% Number of training examples
m = size(X, 1);
error_train = zeros(m, 1);
error_val = zeros(m, 1);

% NOTE: A better way to do this: Randomly selected examples
% For i examples...
% Randomly choose i examples from X and Xval (training + cv set)
% Repeat N times
% Average the errors - store into error_train + error_val
for i = 1:m
    X_i = X(1:i,:);
    y_i = y(1:i);
    [theta] = trainLinearReg(X_i, y_i, lambda); % Get parameters for training size i
    H_i = X_i * theta;
    
    % Use parameters learned to evaluate error on training set size i
    error_train(i) = 1/(2*i)*((H_i-y_i)'*(H_i-y_i));
    
    % Use parameters learned to eval error on WHOLE cross-validation set
    H_val = Xval * theta;
    error_val(i) = 1/(2*length(yval))*((H_val-yval)'*(H_val-yval));
end

end
