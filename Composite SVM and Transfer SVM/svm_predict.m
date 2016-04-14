function [funMargin, predict] = svm_predict(X_train, Y_train, X_test, cost, ChosenKernel, sigma, offset, degree)
% X_train is N*M matrix
% Y_train is N*1 matrix
% X_test is N2*M2 matrix
% cost is the constraint
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter

% Output funMargin is f(x) = W'X + b
% Output predict is 1 if >0 otherwise is -1

[alphas, spt_idx, intercept] = svm_train(X_train, Y_train, cost, ChosenKernel, sigma, offset, degree);
N = size(X_test, 1);
funMargin = zeros(N,1);
predict = zeros(N,1);
for i=1:N
    funMargin(i) = decision_Fun(X_train, X_test(i,:), alphas, Y_train, ChosenKernel, sigma, offset, degree) + intercept;
end
predict(find(funMargin>0))=1;
predict(find(funMargin<0))=-1;
end


