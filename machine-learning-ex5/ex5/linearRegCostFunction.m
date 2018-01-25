function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

m = length(y); % number of training examples

% Initialize J and grad 
J = 0;
grad = zeros(size(theta));

t = theta;
t(1) = 0; % set the first element of theta to 0
h = X * theta; % h(x)

% Cost of regularized linear regression
J = sum((h - y).^2)/(2*m) + (lambda/(2*m))*sum(t.^2);;

% Gradient
grad = (1/m)*X'*(h-y) + (lambda*t)/m;
grad = grad(:);

end
