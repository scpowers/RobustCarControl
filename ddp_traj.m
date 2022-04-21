function xs = ddp_traj(x0, us, S)

N = size(us, 2);
xs(:,1) = x0;

for k=1:N,
  xs(:, k+1) = S.f(k, xs(:,k), us(:,k), S);
end
