function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

idx = zeros(size(X,1), 1);
for i=1:size(X,1) % Go through every example
    min_diff = realmax;
    min_centr = -1;
    for j=1:K % For each example, go through every cluster
        diff = X(i,:)-centroids(j,:); % And find distance to the j-th cluster
        diff = diff*diff';
        if diff < min_diff
            min_diff = diff;
            min_centr = j;
        end
    end
    idx(i) = min_centr;
end

end

