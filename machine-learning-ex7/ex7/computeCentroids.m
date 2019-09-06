function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

centroids = zeros(K, n);

for i=1:K
    logic = idx == i; % Get logical vector of examples assigned to centroid j
    X_cent = X .* logic; % Filter out actual examples with logical vector
    sums = sum(X_cent); % Sum down columns
    avg = sums / sum(idx == i); % Get average
    centroids(i,:) = avg;
end

end

