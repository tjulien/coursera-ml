function A_inv_B = matrixInverseVector( A, b, x_init, alpha )
  x_new = x_init;  
  while norm(A*x_new - b)^2 >= 10^-6
    x_new = x_new - alpha * 2*A*(A*x_new - b);
  end
  A_inv_B = x_new;
end