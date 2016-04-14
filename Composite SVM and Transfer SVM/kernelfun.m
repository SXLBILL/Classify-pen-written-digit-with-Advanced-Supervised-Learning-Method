function K = kernelfun(x, y, ChosenKernel, sigma, offset, degree)
% x and y is N*1 matrix
% ChosenKernel is the kernel choosed (linear, rbf, polynomial)
% sigma is kernel rbf's parameter
% offset and degree is polynomial's parameter
% output K is to return a value

if strcmp(ChosenKernel, 'polynomial');
    K = (sum(x.*y) + offset)^degree;
elseif strcmp(ChosenKernel, 'rbf');
    K = exp(-sum((x-y).^2)/(2*sigma^2));
else strcmp(ChosenKernel, 'linear');
    K = sum(x.*y);
end

end

    