function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% Random permutation (no repeats) of indices of X
randidx = randperm(size(X,1));

% Take first K examples as centroids
centroids = X(randidx(1:K),:);

end

