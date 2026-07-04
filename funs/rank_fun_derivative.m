function y = rank_fun_derivative(x,delta)


yyy=delta*exp(-x.^2);
yy = yyy.*(1+2*delta.*x+2*(x.^2));
y =yy./((x+delta).^2+eps);


end
