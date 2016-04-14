function decision = decision_Fun(X_train, X_unknown, alphas, Y_train, ChosenKernel, sigma, offset, degree)
% X_train is N*M matrix
% X_unkown is a N*1 matrix
% alphas is the SVM dual problem parameters
% Y_train is N*1 matrix
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter

% Output decision is to return a value ( WX )

N = size(X_train, 1);
D = zeros(N, 1);
for i=1:N
    D(i) = alphas(i)*Y_train(i)*kernelfun(X_train(i,:), X_unknown, ChosenKernel, sigma, offset, degree);
end
decision = sum(D);
end

