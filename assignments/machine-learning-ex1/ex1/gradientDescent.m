function theta_history = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
%J_history = zeros(num_iters, 1);
theta_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    % 1. create the hypothesis vector, which is X times theta
    h = X * theta;

    % 2. create the error vector, the difference between the hypothesis vector and y
    
    error = h - y;
    % error is size 97x1

    % 3. calculate the change of theta (which is the sum of the product of X
    %    and the errors vector, scaled by alpha and (1/m)

    ch = alpha * (1/m) * (X' * error);

    % 4. update theta by subtracting the change in theta by the starting value of theta

   theta = theta - ch;
   %size(theta)
   %theta

    % ============================================================

    % Save the cost J in every iteration    
    %J_history(iter) = computeCost(X, y, theta);
    theta_history(iter,1:2) = theta;
    %size(theta_history)
    %theta_history(iter, :)
end

end
