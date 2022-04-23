function f = run_exp()
% Spencer Powers
% Nonlinear Control and Planning in Robotics, Spring 2022
% Final Project
% Ideal trajectories generated via modified form of ddp_car_obst.m by Dr. Kobilarov

% Changelog:
% * Added S.umin and S.umax fields holding max and min control values

clc; clear variables; close all;

%%%%%%%%%%%%%%%%%%%%%% Optimal Trajectory Generation %%%%%%%%%%%%%%%%%%%%%%
% time horizon and segments
tf = 20;
S.N = 80;
S.h = tf/S.N;

% car parameters
S.l = 1; % distance between axles
S.circ_r = 0.5; %radius of circle centered on each axle for collision model

% cost function parameters
S.Q = .0*diag([5, 5, 1, 1]);
S.R = diag([1, 1]);
S.Qf = diag([5, 5, 1, 1]);

S.f = @car_f;
S.L = @car_L;
S.Lf = @car_Lf;
S.mu = 0;

% initial state
% x = [p_x p_y theta phi v]
x0 = [-5; -5; 0; 0];

% desired state
xd = [1; 1; 0; 0];
S.xd = xd;

% define obstacles
S.os(1).p = [-2.5;-2.5];
S.os(1).r = 1;
S.ko = 1e4; % beta coeff in HW7 Q3 problem statement

% add control bounds
S.umin = [-.4, -.4];
S.umax = [.4, .4];

% initial control sequence
us = zeros(2,S.N);

xs = ddp_traj(x0, us, S);

J_init = ddp_cost(xs, us,  S)

subplot(1,2,1)

plot(xs(1,:), xs(2,:), '-b')
hold on

% just drawing circular obstacles
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

plot(xs(1,:), xs(2,:), '-g'); % plot final trajectory

J = ddp_cost(xs, us, S) % final minimized cost

xlabel('x')
ylabel('y')

% plot controls
subplot(1,2,2)
plot(0:S.h:tf-S.h, us(1,:),0:S.h:tf-S.h, us(2,:));
xlabel('sec.')
legend('u_1','u_2')


%%%%%%%%%%%%%%%%%%%%%%%% Tracking via Backstepping %%%%%%%%%%%%%%%%%%%%%%%%
% say you're starting at some offset from the ideal x0
x0_noisy = x0 + [0.1; 0.1; 0.1; 0.1];

% can't just use ideal controls, as those assume you're starting from x0
% but you can use the ideal trajectory for reference (to compute errors)
S.xs = xs;
S.us = us;

% trajectory could still be discretized into N segments, still have same
% number of controls to compute, they'll just be slightly different than
% the ideal ones
x_actual = zeros(size(xs));
u_actual = zeros(size(us));

x_actual(:,1) = x0_noisy; % start at noisy initial state
for i=1:S.N
    u = car_ctrl(x_actual(:,i), S, i); % compute tracking control
    u_actual(:,i) = u;
    x_actual(:, i+1) = S.f(i, x_actual(:,i), u, S); % compute next state
end

% compare the ideal and actual trajectories
figure;
plot(xs(1,:), xs(2,:), '--g'); % plot final trajectory
hold on;
plot(x_actual(1,:), x_actual(2,:), '-g'); % plot final trajectory

% compare the ideal and actual controls
figure;
plot(0:S.h:tf-S.h, us(1,:), '--b');
hold on;
plot(0:S.h:tf-S.h, us(2,:), '--r');
plot(0:S.h:tf-S.h, u_actual(1,:), '-b');
plot(0:S.h:tf-S.h, u_actual(2,:), '-r');


end


% feedback control law derived via backstepping
function u = car_ctrl(x, S, i)
k1 = 1;
k2 = 1;

% get info about the reference trajectory at this particular spot
x_ref = S.xs(:,i);
u_ref = S.us(:,i);
yd = x_ref(1:2); % get yd at this spot
theta_ref = x_ref(3);
v_ref = x_ref(4);

dyd = v_ref*[cos(theta_ref); sin(theta_ref)]; % from dynamics
d2yd = [-v_ref^2*sin(theta_ref)/S.l, cos(theta_ref);
     v_ref^2*cos(theta_ref)/S.l, sin(theta_ref)] * [tan(u_ref(1)); u_ref(2)];

% get info about the current real state
y = x(1:2);
theta = x(3);
v = x(4);

% current velocity
dy = v*[cos(theta); sin(theta)]; % from dynamics

% error states
e = y - yd;
z = -dyd + k1*e + dy;
e_dot = dy - dyd;

% augmented inputs tmp =(tan(u1), u2)
R = [-v^2*sin(theta)/S.l, cos(theta);
     v^2*cos(theta)/S.l, sin(theta)];
tmp = inv(R)*(-k2*z - e + d2yd - k1*e_dot);
u = [atan(tmp(1)); tmp(2)];

%restrict controls to limits
for i=1:2
    if u(i) > S.umax(i)
        u(i) = S.umax(i);
    elseif u(i) < S.umin(i)
        u(i) = S.umin(i);
    else
    end
end

end


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

% quadratic penalty term for collisions
if isfield(S, 'os')
  for i=1:length(S.os) % for each obstacle
    for j = 0:1 % for each collision checking circle (on each axle)
        circ_center = [x(1) + j*S.l*cos(x(3)); x(2) + j*S.l*sin(x(3))];
        g = circ_center - S.os(i).p;
        c = (S.os(i).r + S.circ_r) - norm(g);
        if c < 0
         continue
        end
    
        L = L + S.ko/2*c^2;
        v = g/norm(g);
        Lx(1:2) = Lx(1:2) - S.ko*c*v;
        Lxx(1:2,1:2) = Lxx(1:2,1:2) + S.ko*v*v';  % Gauss-Newton appox
    end
  end
end
end


function [x, A, B] = car_f(k, x, u, S)
% car dynamics and jacobians

dt = S.h;
theta = x(3);
v = x(4);

A = [1 0 -dt*v*sin(theta) dt*cos(theta);
     0 1 dt*v*cos(theta) dt*sin(theta);
     0 0 1 dt*tan(u(1))/S.l;
     0 0 0 1];

B = [0 0;
     0 0;
     dt*v*sec(u(1))^2/S.l 0;
     0 dt];

x = [x(1) + dt*v*cos(theta);
     x(2) + dt*v*sin(theta);
     x(3) + dt*v*tan(u(1))/S.l;
     x(4) + dt*u(2)];
    
end