function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigma_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Variable to store the minumum error
min_error = 0;

for i = 1:length(C_vec)
  C = C_vec(i);
  for j = 1:length(sigma_vec)
    sigma = sigma_vec(j);
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    % This finds the minimum error encountered
    if (error < min_error || (i == 1 && j == 1))
      min_error = error;
      min_C = C;
      min_sigma = sigma;
    endif
  endfor
endfor

C = min_C;
sigma = min_sigma;

end
