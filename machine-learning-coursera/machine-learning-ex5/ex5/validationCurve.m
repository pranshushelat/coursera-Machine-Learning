function [lambda_vec, error_train, error_val] = ...
    validationCurve(X, y, Xval, yval)
%VALIDATIONCURVE Generate the train and validation errors needed to
%plot a validation curve that we can use to select lambda
%   [lambda_vec, error_train, error_val] = ...
%       VALIDATIONCURVE(X, y, Xval, yval) returns the train
%       and validation errors (in error_train, error_val)
%       for different values of lambda. You are given the training set (X,
%       y) and validation set (Xval, yval).
%

% Selected values of lambda (you should not change this)
lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';

% You need to return these variables correctly.
error_train = zeros(length(lambda_vec), 1);
error_val = zeros(length(lambda_vec), 1);

% Number of traning and cross validation examples
m = size(X, 1);
mval = size(Xval, 1);

% Errors for different values of lambda
for i = 1:length(lambda_vec)
  lambda = lambda_vec(i);
  [theta] = trainLinearReg([ones(m, 1) X], y, lambda); % training parameters
  [error_train(i),grad] = linearRegCostFunction...     % training error
                           ([ones(m, 1) X], y, theta, 0);
  [error_val(i),grad] = linearRegCostFunction...       % cross validation error
                           ([ones(mval, 1) Xval], yval, theta, 0);
endfor

end
