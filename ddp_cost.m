function J  = ddp_cost(xs, us, S)

N = size(us, 2);
J = 0;

for k=1:N+1,
  if k < N+1
    [L, Lx, Lxx, Lu, Luu] = S.L(k, xs(:,k), us(:,k), S);
  else
    [L, Lx, Lxx, Lu, Luu] = S.L(N+1, xs(:,end), [], S);  
  end
  J = J + L;
end