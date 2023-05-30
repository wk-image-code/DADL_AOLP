function [G,A]=D_AOLP(D)
%{
G--the graph of the adjacency matrix
A--the adjacency matrix

D--Dictionary
%}
c = linspace(1,10,size(D,1));
K = 3;
% Calculate the Euclidean distance between each sample in the eigenmatrix D
distances = EuDist2(D,[],1);
distances(distances<1E-4)=0;
D_ = distances;
% The KNN algorithm is used to compute the neighbor set
[idx,~] = knnsearch(D_,D_,'k',K);
% Construct the adjacency matrix
A = zeros(size(D_,1));
for i = 1:size(D_,1)
    A(i,idx(i,:)) = 1;
end

for i = 1:size(D_,1)
    A(i,i) = 0;
end
G = graph(A,'lower');




