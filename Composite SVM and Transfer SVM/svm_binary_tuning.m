function [CV_MSE,  FinalPredMatrix, FinalTrueMatrix, FinalScoreMatrix, ConfusionMatrix] = svm_binary_tuning(X_whole, Y_whole, cost, ChosenKernel, sigma, offset, degree)
% X_whole is N*M matrix
% Y_whole is N*1 matrix which includes 1, -1
% cost is the constraint for alphas
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter

% Output CV_MSE is to calculate the correct rate after CV
% Output FinalPredMatrix is a matrix to predict binary SVM which contains -1 or 1
% Output FinalTrueMatrix is a matrix which contains true value for all CV test observations
% Output FinalScoreMatrix is a matrix which contains score for binary SVM
% Output Confusion matrix is a confusion matrix actual vs predict

% Object: The function is to measure a binary classification SVM
% model's performance by calculate the cv mse

% k-fold cross validation
k = 5;
cvFolds = crossvalind('Kfold', X_whole(:,1), k);
classNum = 1;
% intial pred,score and true matrix for CV
FinalScoreMatrix = zeros(size(X_whole, 1), classNum);
FinalTrueMatrix = zeros(size(X_whole, 1), classNum);
FinalPredMatrix = zeros(size(X_whole, 1), classNum);
ConfusionMatrix = zeros(2, 2);
StartPoint = 0;
EndPoint = 0;

for j = 1:k
    disp(j)
    % split test data and training data
    testIdx = (cvFolds == j);
    trainIdx = ~testIdx;
    Y_train = Y_whole(trainIdx);
    Y_test = Y_whole(testIdx);
    X_train = X_whole(trainIdx,:);
    X_test = X_whole(testIdx,:);
    trainNum = length(Y_train);
    testNum = length(Y_test);
    
    % generate the real matrix
    M_real = Y_test;

    % make a score matrix for a certain test dataset
    pred_score_matrix = zeros(testNum, classNum);
    pred_matrix = zeros(testNum, classNum);
    
    % test using test instances
    [funMargin, predict] = svm_predict(X_train, Y_train, X_test, cost, ChosenKernel, sigma, offset, degree);
    % make the 1's score be positive and 0's score be negative
    pred_score_matrix = funMargin;
    pred_matrix = predict;
  
    % Create the final score Matrix and true matrix
    StartPoint = StartPoint + 1;
    EndPoint = StartPoint + testNum - 1;
    FinalScoreMatrix(StartPoint:EndPoint,:) = pred_score_matrix;
    FinalTrueMatrix(StartPoint:EndPoint,:) = M_real;
    FinalPredMatrix(StartPoint:EndPoint,:) = pred_matrix;
    StartPoint = EndPoint;
end
FinalScoreMatrix;
FinalTrueMatrix;
FinalPredMatrix;

% get k-fold generalization error
TP = 0;
FP = 0;
FN = 0;
TN = 0;
for i=1:size(FinalTrueMatrix, 1)
    if FinalPredMatrix(i) == 1
        if FinalTrueMatrix(i)==1
            TP = TP + 1;
        else
            FP = FP + 1;
        end
    else
        if FinalTrueMatrix(i) ==1
            FN = FN + 1;
        else
            TN = TN + 1;
        end
    end
end

CV_MSE = 1 - (TP+ TN)/(TP+FP+FN+TN);
ConfusionMatrix(1,1) = TP
ConfusionMatrix(2,1) = FP
ConfusionMatrix(1,2) = FN
ConfusionMatrix(2,2) = TN
end