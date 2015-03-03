M = dlmread('ex6DataPrepared/train-features-50.txt', ' ');
spmatrix = sparse(M(:,1), M(:,2), M(:,3),700,2500);
train_matrix = full(spmatrix);
train_labels = dlmread('ex6DataPrepared/train-labels-50.txt');

numTrainDocs = size(train_matrix,1);
numToken = size(train_matrix,2);

spam = train_matrix(train_labels==1,:);
non_spam = train_matrix(train_labels==0,:);

length_spam = sum(spam,2);
length_non_spam = sum(non_spam,2);

lh_spam = (sum(spam,1)+1)./(sum(length_spam,1)+numToken);
lh_non_spam = (sum(non_spam,1)+1)./(sum(length_non_spam,1)+numToken);

prior_spam = size(spam,1)/numTrainDocs;
prior_non_spam = 1 - prior_spam;

T = dlmread('ex6DataPrepared/test-features.txt', ' ');
spmatrix2 = sparse(T(:,1), T(:,2), T(:,3));
test_matrix = full(spmatrix2);
test_labels = dlmread('ex6DataPrepared/test-labels.txt');
numTestDocs = size(test_matrix,1);

post_spam = test_matrix*log(lh_spam')+prior_spam;
post_non_spam = test_matrix*log(lh_non_spam')+prior_non_spam;

result = post_spam > post_non_spam;
numdocs_wrong = sum(xor(result, test_labels))
hit_rate = 1 - numdocs_wrong/numTestDocs
