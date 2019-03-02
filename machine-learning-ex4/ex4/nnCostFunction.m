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

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
a1 = [ones(m, 1) X];

z2 = a1*Theta1';

sig = sigmoid (a1*Theta1');

sig = [ones(m,1) sig];

h = sigmoid(sig*Theta2');

s= zeros(m, num_labels);

for i=1:m,
  e=s(i,:);
  e(y(i))=1;
  s(i,:)=e;
  i=i+1;
end;
  
  r=0;
  
for i=1:m,
   w=h(i,:)';
   c=s(i,:);
   r = r + (((-c)*(log(w)))-((1-c)*(log(1-w))));
   i = i+1;
end;

   J = r/m;
   
   row=0;
   ro=0;
   
for i=1:hidden_layer_size,
  col = Theta1(i,:)';
  col = ((sum(col.^2))-((col(1,1))^2));
  row = row + col;
  i = i + 1;
end;

for i=1:num_labels,
  col = Theta2(i,:)';
  col = ((sum(col.^2))-((col(1,1))^2));
  ro = ro + col;
  i = i+1;
end; 

row = row + ro;

reg = ((lambda/(2*m))*row);

J = J + reg; 
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

sigm3 = h.-s;
sigm2 = (sigm3*Theta2).*sigmoidGradient([ones(size(z2,1),1) z2]);
sigm2 = sigm2(: , 2:end);

Theta1_grad = ((sigm2'*a1)./m);
Theta2_grad = ((sigm3'*sig)./m);

regul1 = (lambda/m)*[zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
regul2 = (lambda/m)*[zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

Theta1_grad = Theta1_grad + regul1;
Theta2_grad = Theta2_grad + regul2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
