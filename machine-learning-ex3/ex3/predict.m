function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
X = [ones(m, 1) X]; % add column of ones to beginning - bias unit (+1)
% each row is a separate NN, layer 1 vector: [x_0, x_1, x_2, ..., x_400]

% each row is a separate NN, layer 2 vector: [a_0, a_1, a_2, ..., a_25]
A2 = sigmoid(X*Theta1');
A2 = [ones(size(A2,1), 1) A2]; % add bias unit (+1)
[m, p] = max(sigmoid(A2*Theta2'), [], 2);

end
