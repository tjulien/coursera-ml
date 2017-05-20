function cost = squaredError( A, b, x )
%SQUAREDERROR Summary of this function goes here
%   Detailed explanation goes here
cost = norm(A*x - b)^2;

end

