function [pred_score_matrix, pred_matrix, FinalPredMatrix]=svm_predict_composite(X_train, Y_train, X_test, cost, ChosenKernel, sigma, offset, degree)
unique_class = unique(Y_train);
class_num = length(unique_class);
pred_score_matrix = zeros(size(X_test,1), class_num);
pred_matrix = zeros(size(X_test,1), class_num);
FinalPredMatrix = zeros(size(X_test,1), 1);

for i = 1:class_num
    disp(i)
    Class_groups = ismember(Y_train, unique_class(i))*2 - 1;
    % test using test instances
    [funMargin, predict] = svm_predict(X_train, Class_groups, X_test, cost, ChosenKernel, sigma, offset, degree);
    % make the 1's score be positive and 0's score be negative
    pred_score_matrix(:, i) = funMargin;
    pred_matrix(:, i) = predict;
end

for i = 1:size(X_test,1)
    predMaxIdx = find(pred_score_matrix(i,:)== max(pred_score_matrix(i,:)))-1;
    FinalPredMatrix(i) = predMaxIdx ;
end

end

