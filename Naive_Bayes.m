

numTokens = 2500;
train_labels = dlmread('ex6DataPrepared/train-labels.txt');
numTrainDocs = size(train_labels,1);
M = dlmread('ex6DataPrepared/train-features.txt', ' ');
spmatrix = sparse(M(:,1), M(:,2), M(:,3), numTrainDocs, numTokens);
train_matrix = full(spmatrix);


non_spam = zeros(1,numTokens);
spam = zeros(1,numTokens);
for i=1:numTrainDocs
    if train_labels(i)==0
        non_spam = [non_spam;train_matrix(i,:)];
    else
        spam = [spam;train_matrix(i,:)];
    end
end
non_spam(1,:)=[];
spam(1,:)=[];


lh_spam = (sum(spam)+1)/(size(spam,1)+2500);
lh_non_spam = (sum(non_spam)+1)/(size(non_spam,1)+2500);

prior_spam = size(spam:1)/(size(train_matrix:1)+2500);
prior_non_spam = size(non_spam:1)/size(train_matrix:1);

test_labels = dlmread('ex6DataPrepared/test-labels.txt');
numTestDocs = size(test_labels,1);
T = dlmread('ex6DataPrepared/test-features.txt', ' ');
spmatrix2 = sparse(T(:,1), T(:,2), T(:,3), numTestDocs, numTokens);
test_matrix = full(spmatrix2);

post_spam = zeros(numTestDocs,1);
post_non_spam = zeros(numTestDocs,1);

for i =1:numTestDocs
    post_spam(i)=log(prior_spam);
    post_non_spam(i)=log(prior_non_spam);
    for j = 1:numTokens
        post_spam(i)= post_spam(i) + test_matrix(i,j)*log(lh_spam(j));
        post_non_spam(i)=post_non_spam(i) + test_matrix(i,j)*log(lh_non_spam(j));
    end
end

result = post_spam - post_non_spam;
g = find(result>0);
f = find(result<=0);
result(g)=1;
result(f)=0;
(numTestDocs-sum(result-test_labels))/numTestDocs



