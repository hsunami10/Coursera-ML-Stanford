function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

C = 1;
sigma = 0.3;
best_error = size(X,1);

steps = [0.01 0.03 0.1 0.3 1 3 10 30]';

% Iterate over all possible combinations of C and sigma to find parameters
% that give the least error in the cross validation set
for ci=1:length(steps)
    for si=1:length(steps)
        model = svmTrain(X, y, steps(ci), @(x1, x2) gaussianKernel(x1, x2, steps(si)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        if error < best_error
            C = steps(ci);
            sigma = steps(si);
            best_error = error;
        end
    end
end

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
