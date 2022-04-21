function f = run()
% Spencer Powers
% Applied Optimal Control, Fall 2021
% Modified form of ddp_car_obst.m by Dr. Kobilarov

% Changelog:
% * Added S.umin and S.umax fields holding max and min control values


clc; clear variables; close all;
% time horizon and segments
tf = 20;
S.N = 32;
S.h = tf/S.N;

% car parameters
S.l = 1;

% cost function parameters
S.Q = .0*diag([5, 5, 1, 1, 1]);
S.R = diag([1, 1]);
S.Qf = diag([5, 5, 1, 1, 1]);

S.f = @car_f;
S.L = @car_L;
S.Lf = @car_Lf;
S.mu = 0;

% initial state
% x = [p_x p_y theta phi v]
x0 = [-5; -5; 0; 0; 0];

% desired state
xd = [1; 1; 0; 0; 0];
S.xd = xd;

S.os(1).p = [-2.5;-2.5];
S.os(1).r = 1;
S.ko = 1e4; % beta coeff in HW7 Q3 problem statement

% add control bounds
S.umin = [-.1, -.1];
S.umax = [.1, .1];

% initial control sequence
us = zeros(2,S.N);

xs = ddp_traj(x0, us, S);

J_init = ddp_cost(xs, us,  S)

subplot(1,2,1)

plot(xs(1,:), xs(2,:), '-b')
hold on

if isfield(S, 'os')
  da = .1;
  a = -da:da:2*pi;
  for i=1:length(S.os)
    % draw obstacle
    plot(S.os(i).p(1) + cos(a)*S.os(i).r,  S.os(i).p(2) + sin(a)*S.os(i).r, ...
         '-r','LineWidth',2);
  end
  axis equal
end



S.a = 1;

for i=1:50
  [dus, V, Vn, dV, a] = ddp(x0, us, S);

  % update controls
  us = us + dus;
  
  S.a = a;   % reuse step-size for efficiency
  
  % update trajectory
  xs = ddp_traj(x0, us, S);

  plot(xs(1,:), xs(2,:), '-b');
end

plot(xs(1,:), xs(2,:), '-g');

J = ddp_cost(xs, us, S)

xlabel('x')
ylabel('y')

subplot(1,2,2)

plot(0:S.h:tf-S.h, us(1,:),0:S.h:tf-S.h, us(2,:));
xlabel('sec.')
legend('u_1','u_2')



function [L, Lx, Lxx, Lu, Luu] = car_L(k, x, u, S)
% car cost (just standard quadratic cost)

if (k == S.N+1)
  if isfield(S, 'xd')
        xfError = x - S.xd; 
  else
        xfError = x;
  end
  L = xfError'*S.Qf*xfError/2;
  Lx = S.Qf*xfError;
  Lxx = S.Qf;
  Lu = [];
  Luu = [];
else
  L = S.h/2*(x'*S.Q*x + u'*S.R*u);
  Lx = S.h*S.Q*x;
  Lxx = S.h*S.Q;
  Lu = S.h*S.R*u;
  Luu = S.h*S.R;
end

% quadratic penalty term
if isfield(S, 'os')
  for i=1:length(S.os)
    g = x(1:2) - S.os(i).p;
    c = S.os(i).r - norm(g);
    if c < 0
      continue
    end
    
    L = L + S.ko/2*c^2;
    v = g/norm(g);
    Lx(1:2) = Lx(1:2) - S.ko*c*v;
    Lxx(1:2,1:2) = Lxx(1:2,1:2) + S.ko*v*v';  % Gauss-Newton appox
  end
end


function [x, A, B] = car_f(k, x, u, S)
% car dynamics and jacobians

dt = S.h;
theta = x(3);
phi = x(4);
v = x(5);


A = [1 0 -dt*v*sin(theta)*cos(phi) -dt*v*cos(theta)*sin(phi) dt*cos(theta)*cos(phi);
     0 1 dt*v*cos(theta)*cos(phi) -dt*v*sin(theta)*sin(phi) dt*sin(theta)*cos(phi);
     0 0 1 dt*v*cos(phi)/S.l dt*sin(phi)/S.l;
     0 0 0 1 0;
     0 0 0 0 1];

B = [0 0;
     0 0;
     0 0;
     dt 0;
     0 dt];

x = [x(1) + dt*v*cos(theta)*cos(phi);
     x(2) + dt*v*sin(theta)*cos(phi);
     theta + dt*v*sin(phi)/S.l;
     phi + dt*u(1);
     v + dt*u(2)];