function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    h = X * theta; % calculate hypotheses
    diff = h - y; % take difference between hypotheses and actual values
    
    % calculate derivates - multiply by Xj(i), sum all, divide by m
    jDeriv1 = sum((diff .* X(:,1))) / m;
    jDeriv2 = sum((diff .* X(:,2))) / m;
    
    % calculate corresponding thetas
    theta(1) = theta(1) - (alpha * jDeriv1);
    theta(2) = theta(2) - (alpha * jDeriv2);

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);
end

end
