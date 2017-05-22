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

X = [ones(m, 1) X];

for i = 1:m
    y_k_i = zeros(num_labels, 1);
    y_k_i(y(i)) = 1;

    z_2 = Theta1 * X(i, :)';
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];

    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);

    h_k_i = a_3;
        
    J = J + ((-y_k_i' * log(h_k_i)) - ((1 - y_k_i)' * log(1 - h_k_i)));                                                                                                                                
end

J = J / m;

reg = 0;
for j = 1:size(Theta1, 1)
    for k = 2:size(Theta1, 2)
        reg = reg + Theta1(j, k) ^ 2;
    end
end

for j = 1:size(Theta2, 1)
    for k = 2:size(Theta2, 2)
        reg = reg + Theta2(j, k) ^ 2;
    end
end

reg = reg * (lambda / (2 * m));

J = J + reg;

%
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

for t = 1:m
    z_2 = Theta1 * X(t, :)';
    a_2 = sigmoid(z_2);
    a_2 = [1; a_2];

    z_3 = Theta2 * a_2;
    a_3 = sigmoid(z_3);
    
    y_k = zeros(num_labels, 1);
    y_k(y(t)) = 1;
    
    d_3 = (a_3 - y_k);
    
    d_2 = Theta2' * d_3;
    d_2 = d_2(2:end) .* sigmoidGradient(z_2);
    Theta2_grad = Theta2_grad + d_3 * a_2';
    Theta1_grad = Theta1_grad + d_2 * X(t, :);
end

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


reg_Theta1 = Theta1;
for i = 1:size(reg_Theta1, 1)
    reg_Theta1(i, 1) = 0;
end

reg_Theta1 = reg_Theta1 .* (lambda / m);
Theta1_grad = Theta1_grad + reg_Theta1;

reg_Theta2 = Theta2;
for i = 1:size(reg_Theta2, 1)
    reg_Theta2(i, 1) = 0;
end

reg_Theta2 = reg_Theta2 .* (lambda / m);
Theta2_grad = Theta2_grad + reg_Theta2;
















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
