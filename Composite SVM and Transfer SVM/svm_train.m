function [alphas, spt_idx, intercept] = svm_train(X_train, Y_train, cost, ChosenKernel, sigma, offset, degree)
% X_train is N*M matrix
% Y_train is N*1 matrix
% cost is the constraint
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter

% alphas is the dual problem solution
% spt_idx is the index of support vectors
% intercept is the w0

% calculate the observation number
N = size(X_train, 1);

% initial kernel matrix
if strcmp(ChosenKernel, 'rbf');
    Kmatrix = zeros(N, N);
    for i = 1:N
        for j = 1:N
            Kmatrix(i,j) = kernelfun(X_train(i,:), X_train(j,:), ChosenKernel, sigma, offset, degree);
        end
    end
elseif strcmp(ChosenKernel, 'polynomial');
    Kmatrix = (X_train*X_train' + offset).^degree;
else
    Kmatrix = X_train*X_train';
end
% quadratic programing to solve dual problem of SVM
options = optimset('LargeScale','on','MaxIter',2000);
eps = 1e-4;
Y = Y_train*Y_train';
H = Kmatrix.*Y + eps.*eye(N);
f = -1.*ones(N, 1);
Aeq = Y_train';
beq = 0;
LB = zeros(N, 1);
UB = cost.*ones(N, 1);
alphas = quadprog(H, f, [], [], Aeq, beq, LB, UB,[],options);
spt_idx = find(alphas > 1e-4);

% compute the intercept w0 (Andrew Ng svm mentioned that)
idx_neg = find(Y_train==-1);
idx_poz = find(Y_train==1);
neg_d = zeros(size(idx_neg, 1), 1);
poz_d = zeros(size(idx_poz, 1), 1);
for i = 1:size(idx_neg, 1)
    neg_d = decision_Fun(X_train, X_train(idx_neg(i),:), alphas, Y_train, ChosenKernel, sigma, offset, degree);
end
for i = 1:size(idx_poz, 1)
    poz_d = decision_Fun(X_train, X_train(idx_poz(i),:), alphas, Y_train, ChosenKernel, sigma, offset, degree);
end
intercept = -(max(neg_d) + min(poz_d))/2;

end







