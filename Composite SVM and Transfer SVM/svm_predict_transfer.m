function [funMargin, predict] = svm_predict_transfer(X_train_old, Y_train_old, X_train_new, Y_train_new, X_test_new, cost1, cost2, ChosenKernel, sigma, offset, degree)
% X_train_old is N*M matrix which belongs to source data
% Y_train_old is N*1 matrix which belongs to source data
% X_train_new is N2*M matrix which belongs to target data
% Y_train_new is N2*1 matrix which belongs to target data
% X_test_new is N3*M matrix which belongs to unlabeled target data
% cost1 is the constraint for source data dual problem
% cost2 is the constraint for target data dual problem
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter

% funMargin is the distance to hyperplane with direction
% predict is 1 or -1; if funMargin >0 then predict 1, otherwise -1

% Calculate the source data alphas
[alphas_old, spt_idx_old, intercept_old] = svm_train(X_train_old, Y_train_old, cost1, ChosenKernel, sigma, offset, degree);

% Calculate fs(x)=sum(alpha_old*y*xs*x)
old_decision_fun = zeros(size(X_train_new,1), 1);
for i=1:size(X_train_new, 1)
    old_decision_fun(i) = decision_Fun(X_train_old, X_train_new(i,:), alphas_old, Y_train_old, ChosenKernel, sigma, offset, degree) ;
end

% Dual problem has changed for transfer svm
lambda = Y_train_new.*old_decision_fun;
f = lambda - 1;
% Calculate the target data alphas
alphas_new = svm_train_transfer(f, X_train_new, Y_train_new, cost2, ChosenKernel, sigma, offset, degree);

% ft(x) = fs(x) + sum(alpha_new*y*xt*x)
new_decision_partfun = zeros(size(X_train_new,1), 1);
for i=1:size(X_train_new, 1)
    new_decision_partfun(i) = decision_Fun(X_train_new, X_train_new(i,:), alphas_new, Y_train_new, ChosenKernel, sigma, offset, degree); % kernel is the old one
end
new_decision_fun = old_decision_fun + new_decision_partfun;

% compute the intercept w0 (Andrew Ng)
idx_neg = find(Y_train_new==-1);
idx_poz = find(Y_train_new==1);
intercept_new = -(max(new_decision_fun(idx_neg)) + min(new_decision_fun(idx_poz)))/2;

% predict
funMargin = zeros(size(X_test_new,1),1);
predict = zeros(size(X_test_new,1),1);
for i=1:size(X_test_new, 1)
    funMargin(i) = decision_Fun(X_train_old, X_test_new(i,:), alphas_old, Y_train_old, ChosenKernel, sigma, offset, degree) + decision_Fun(X_train_new, X_test_new(i,:), alphas_new, Y_train_new, ChosenKernel, sigma, offset, degree) + intercept_new;
end
predict(find(funMargin>0))=1;
predict(find(funMargin<0))=-1;

end

