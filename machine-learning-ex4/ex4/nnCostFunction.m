function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% Transform the output matrix into suitable for classification
yt = zeros(m,num_labels);
for i = 1:m
  yt(i,y(i)) = 1;
endfor

% Removing the bias terms for Theta1 and Theta2 for regularization
ThetaR1 = Theta1(:,2:end);
ThetaR2 = Theta2(:,2:end);

% Forward Propagation
a1 = [ones(m, 1) X];
z2 = a1*Theta1';
a2 = [ones(m,1) sigmoid(z2)];
z3 = a2*Theta2';
h = sigmoid(z3);

J = (1/m)*sum(sum(-yt.*log(h) - (1-yt).*log(1-h),2))...
           + (lambda/(2*m))*(sum(sum(ThetaR1.^2,2)) + sum(sum(ThetaR2.^2,2)));
           
% Backpropagation algorithm for error terms

D1 = Theta1_grad;
D2 = Theta2_grad;

for t = 1:m

	a1 = [1; X(t,:)'];
	z2 = Theta1 * a1;
	a2 = [1; sigmoid(z2)];
	z3 = Theta2 * a2;
	a3 = sigmoid(z3);

	d3 = a3 - yt(t,:)';

	d2 = Theta2'*d3.*[1; sigmoidGradient(z2)];
	d2 = d2(2:end);

	D1 = D1 + d2*a1';
	D2 = D2 + d3*a2';

end

Theta1_grad = D1/m;
Theta2_grad = D2/m;

Theta1_grad = Theta1_grad + (lambda/m)*([zeros(size(Theta1,1),1) ThetaR1]);
Theta2_grad = Theta2_grad + (lambda/m)*([zeros(size(Theta2,1),1) ThetaR2]);

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end