function [CV_MSE,  FinalPredMatrix, FinalTrueMatrix, FinalScoreMatrix, ConfusionMatrix] = svm_composite_tuning(X_whole, Y_whole, cost, ChosenKernel, sigma, offset, degree)
% X_whole is N*M matrix
% Y_whole is N*1 matrix
% cost is the constraint
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter

% Output CV_MSE is to calculate the correct rate after CV
% Output FinalPredMatrix is a matrix to predict each binary SVM which contains -1 or 1
% Output FinalTrueMatrix is a matrix which contains true value for all CV test observations
% Output FinalScoreMatrix is a matrix which contains score for each binary SVM

% Object: The function is to measure a multi-class classification SVM
% model's performance by calculate the cv mse

seed = 10;
RandStream.setDefaultStream(RandStream('mt19937ar','seed', seed));
% k-fold cross validation
k = 5;
cvFolds = crossvalind('Kfold', X_whole(:,1), k);
Class_unique = unique(Y_whole);
classNum = length(Class_unique); 
% intial pred,score and true matrix for CV
FinalScoreMatrix = zeros(size(X_whole, 1), classNum);
FinalTrueMatrix = zeros(size(X_whole, 1), classNum);
FinalPredMatrix = zeros(size(X_whole, 1), classNum);
ConfusionMatrix = zeros(classNum, classNum);
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
    M_real = zeros(testNum,classNum);
    for i_2 = 1:testNum
        j_2 = find(Class_unique == Y_test(i_2));
        M_real(i_2,j_2) = 1;
    end
    
    % make a score matrix for a certain test dataset
    pred_score_matrix = zeros(testNum, classNum);
    pred_matrix = zeros(testNum, classNum);
    
    for i_1 = 1:classNum
        Class_groups = ismember(Y_train, Class_unique(i_1))*2 - 1;
        % test using test instances
        [funMargin, predict] = svm_predict(X_train, Class_groups, X_test, cost, ChosenKernel, sigma, offset, degree);
        % make the 1's score be positive and 0's score be negative
        pred_score_matrix(:, i_1) = funMargin;
        pred_matrix(:, i_1) = predict;
    end
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
correct = 0;
for i_4 = 1:size(FinalTrueMatrix,1)
    predMaxIdx = find(FinalScoreMatrix(i_4,:)== max(FinalScoreMatrix(i_4,:)));
    predOtherIdx = find(FinalScoreMatrix(i_4,:)~= max(FinalScoreMatrix(i_4,:)));
    tureMaxIdx = find(FinalTrueMatrix(i_4,:)==1);
    ConfusionMatrix(tureMaxIdx, predMaxIdx) = ConfusionMatrix(tureMaxIdx, predMaxIdx) + 1;
    FinalPredMatrix(i_4, predOtherIdx) = 0;
    FinalPredMatrix(i_4, predMaxIdx) = 1;
    if FinalTrueMatrix(i_4, predMaxIdx)==1
        correct = correct + 1;
    end
end
CV_MSE = 1 - correct/size(FinalTrueMatrix,1);
end