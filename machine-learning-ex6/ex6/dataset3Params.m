function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

params = [.01, .03, .1, .3, 1, 3, 10, 30];
cost = inf();
for i = 1:length(params)
    for j = 1:length(params)
        new_C = params(i);
        new_Sigma = params(j);
        model = svmTrain(X, y, new_C, @(x1, x2) gaussianKernel(x1, x2, new_Sigma));
        predictions = svmPredict(model, Xval);
        new_cost = mean(double(predictions ~= yval));
        if new_cost < cost
            cost = new_cost;
            C = new_C;
            sigma = new_Sigma;
        end
    end
end
        




% =========================================================================

end
