function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
J = 0;

% Calculate cost of one forward propagation
X = [ones(m, 1) X]; % add bias unit
A2 = sigmoid(X * Theta1');
A2 = [ones(size(A2, 1), 1), A2]; % add bias unit
H = sigmoid(A2 * Theta2'); % get all hypotheses for all 5000 examples
H_trans = H';

Y = repmat(1:num_labels, m, 1); % create matrix that holds all y vectors (horizontal)
Y = (Y == y)'; % convert into logical matrix (vertical - transpose)
J = sum(sum(-Y.*log(H_trans)-(1-Y).*log(1-H_trans))) / m;

% Calculate regularization term - ignore bias - index from "2"
reg = (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2))+sum(sum(Theta2(:,2:end).^2)));

% Calculation regularized cost
J = J + reg;

Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

for t=1:m
    y = Y(:,t);
    
    % Forward propagate
    a1 = X(t,:)'; % get t-th training example, transpose to column vector - bias unit is already added above
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)]; % add bias unit
    z3 = Theta2 * a2;
    a3 = sigmoid(z3);
    
    % Calculate deltas
    delta_3 = a3 - y;
    delta_2 = (Theta2' * delta_3).*sigmoidGradient([1; z2]); % add 1 here, but not to z2 above because want 1 to be ignored by sigmoid activation function
    delta_2 = delta_2(2:end);
    
    % Calculate and store gradients
    Theta2_grad = Theta2_grad + (delta_3 * a2');
    Theta1_grad = Theta1_grad + (delta_2 * a1');
end

% Take average across all examples
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Add regularization term, ignoring first theta term
Theta1(:,1) = 0;
Theta2(:,1) = 0;
Theta1_grad = Theta1_grad + (lambda/m)*Theta1;
Theta2_grad = Theta2_grad + (lambda/m)*Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
