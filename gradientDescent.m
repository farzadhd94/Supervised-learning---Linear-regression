function [theta, J_history] = gradientDescent(X,y,theta, alpha, num_iters)
%   GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    a=theta(1,1) - alpha*(1/m)*sum((X*theta-y).*X(:,1));
    b=theta(2,1) - alpha*(1/m)*sum((X*theta-y).*X(:,2));
    theta=[a,b]'


    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end
figure(1);
plot(X(:,2), y, 'rx', 'MarkerSize', 10)
hold on;
plot(X*theta)
title('linear regression')
xlabel('value of x')
ylabel('value of y')
legend('Training data', 'Linear regresion')

figure(2);
plot(J_history)
title('cost function');
xlabel('number of iterations');
ylabel('cost function');
end
