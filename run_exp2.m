function f = run_exp2()
% Spencer Powers
% Nonlinear Control and Planning in Robotics, Spring 2022
% Final Project
% Ideal trajectories generated via modified form of ddp_car_obst.m by 
% Dr. Kobilarov

clc; clear variables; close all;

%%%%%%%%%%%%%%%%%%%%%% Optimal Trajectory Generation %%%%%%%%%%%%%%%%%%%%%%
% time horizon and segments
tf = 20;
S.N = 100;
S.h = tf/S.N;

% car parameters
S.l = 1; % distance between axles
S.circ_r = 0.5; %radius of circle centered on each axle for collision model

% generally: ||added noise|| <= S.k_tot*|velocity|
% first noise coefficient: realistically adding noise to tan(u1). So, since
% I'm limiting u1 to +- pi/4, the worst case difference in tan(u1) if you
% have a 3 deg drift while moving at 60 mph is tan(45) - tan(42) = 0.099
% because of the increasingly steep nature of the tangent function...but
% to actually get the coefficient you'd divide 0.099 by 60 mph but in m/s
% if you're moving slower and at a lower steering angle then the difference
% will be smaller, so this is an upper bound.
% second noise coefficient: adding noise directly to u2, so if you are
% moving at 60 mph and wind drops your acceleration by 0.5 m/s^2, then the
% upper bound coefficient is 0.0186 (because of conversion from mph to m/s)
max_drift_deg = 3; % max noise in u1 (in degrees) at 60 mph or 26.8224 m/s
max_drift_acc = 0.5; % max noise in u2 at 60 mph or 26.8224 m/s
k_noise_u1 = (tand(45) - tand(45-max_drift_deg)) / 26.8224;
k_noise_u2 = max_drift_acc / 26.8224;
S.k_noise = [k_noise_u1; k_noise_u2];
S.k_tot = norm(S.k_noise);

% cost function parameters
S.Q = .0*diag([5, 5, 1, 1]); % no accrued cost from states over trajectory
S.R = 4*diag([1, 1]); % penalties on controls over the trajectory
S.Qf = 10*diag([5, 5, 5, 5]); % terminal cost on final state errors

S.f = @car_f; % car dynamics 
S.L = @car_L; % car cost
S.Lf = @car_Lf; % car terminal cost
S.mu = 0;

% initial state
% x = [p_x p_y theta v]
x0 = [-6; -4; 0; 0];

% desired state
xd = [5; -1; 0; 0];
S.xd = xd;

% define obstacles
S.os(1).p = [1.5;-2.5];
S.os(1).r = 1;

% define boundary lines of exclusion zones
% takes the form [x, y, 1]*[coeffs] {sign} 0
S.ez(1).coeffs = [-0.3; 1; 5.5];
S.ez(1).sign = ">=";

S.ko = 1e4; % coeff on cost associated with obstacle collision

% add control bounds
% u = [steering angle; forward acceleration]
S.umin = [-pi/4, -1];
S.umax = [pi/4, 1];

% initial control sequence
us = zeros(2,S.N);
% resulting trajectory from this initial control sequence
xs = ddp_traj(x0, us, S);
% resulting total trajectory cost from this initial control sequence
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

% calling 50 iterations of DDP to optimize the trajectory
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

% plot exclusion zone boundary
if isfield(S, 'ez')
    for i=1:length(S.ez)
        plot([xs(1,1), xs(1,end)+S.l],[ -[S.ez(i).coeffs(1), ...
            S.ez(i).coeffs(3)]*[xs(1,1);1], -[S.ez(i).coeffs(1), ...
            S.ez(i).coeffs(3)]*[xs(1,end)+S.l;1] ], '--r');
    end
end

% show axle circles at start and end
X = [xs(1,1); 
    xs(1,1) + S.l*cos(xs(3,1));
    xs(1,end); 
    xs(1,end) + S.l*cos(xs(3,end))];
Y = [xs(2,1); 
    xs(2,1) + S.l*sin(xs(3,1));
    xs(2,end); 
    xs(2,end) + S.l*sin(xs(3,end))];
R = S.circ_r*ones(4,1);
viscircles([X Y], R, 'Color', 'k', 'LineStyle', '--');

xlabel('x')
ylabel('y')
title('Trajectory Generation')

J = ddp_cost(xs, us, S) % final minimized cost

% plot controls
subplot(1,2,2)
plot(0:S.h:tf-S.h, us(1,:),0:S.h:tf-S.h, us(2,:));
xlabel('sec.')
legend('u_1','u_2')
title('Ideal Controls')


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
% first find the nearest state in the reference trajectory (probably
% won't be the first anymore) and make sure that it's ahead of the noisy
% initial state so we don't start by going backwards. Can do this by 
% having the noisy initial state have some nonzero velocity vaguely pointed
% in the right direction and checking the dot product of this velocity
% vector and the vector from the noisy position to the reference trajectory
% state. This dot product must be > 0 (roughly aligned)
start_index = S.N;
start_dist = norm(x0_noisy(1:2) - S.xs(1:2,S.N));
x0_vel = x0_noisy(4)*[cos(x0_noisy(3)); sin(x0_noisy(3))];
for i=1:S.N-1
    tmp_dist = norm(x0_noisy(1:2) - S.xs(1:2,i));
    vec_to_state = [S.xs(1,i) - x0_noisy(1); S.xs(2,i) - x0_noisy(2)];
    tmp_check = dot(x0_vel, vec_to_state);
    if tmp_dist < start_dist && tmp_check > 0
        start_index = i;
        start_dist = tmp_dist;
    end
end

x_actual(:,start_index) = x0_noisy; % hasn't moved until start_index

% now actually simulate the perturbed system starting at the starting
% index found above
for i=start_index:S.N
    u = car_ctrl(x_actual(:,i), S, i); % compute tracking control
    % add noise proportional to the car's velocity
    % such that ||delta|| <= S.k_tot*abs(velocity)
    % NOTE: u_noise = [tan(u1_noise); u2_noise]...not directly adding to u1
    % general form: |v|*(random uniform number in [-k_i, k_i])
    u_noise = abs(x_actual(4)) * ([-S.k_noise(1); -S.k_noise(2)] + ...
        2*[S.k_noise(1), 0; 0, S.k_noise(2)] * rand(2,1));
    u_noise = [atan(u_noise(1)); u_noise(2)]; % switch back to [u1;u2]
    u = u + u_noise;
    % update storage matrices
    u_actual(:,i) = u;
    x_actual(:, i+1) = S.f(i, x_actual(:,i), u, S); % compute next state
end

% compute tracking error and dist to nearest obstacle over time 
% for performance plots
num_slots = size(x_actual,2) - start_index;
tracking_error = zeros(1, num_slots);
dist_to_obs = zeros(1, num_slots);
for i=0:(num_slots-1)
    % positional error computation
    dist = norm(x_actual(1:2,i+start_index) - S.xs(1:2,i+start_index));
    tracking_error(i+1) = dist;
    
    % nearest dist to obstacle computation
    % right now only working with one obstacle
    % check both circles centered at the axles
    dist = 1e6; % arbitrarily large starting dist, will be replaced in loop
    x = x_actual(1:3, i+start_index);
    if isfield(S, 'os')
        for k=1:length(S.os) % for each obstacle
            for j=0:1
                circ_center = [x(1) + j*S.l*cos(x(3)); x(2) + j*S.l*sin(x(3))];
                g = circ_center - S.os(k).p;
                c = norm(g) - (S.os(k).r + S.circ_r);
                if (c < dist)
                    dist = c; % update closest computed distance to obstacle
                end
            end
        end
    end
    dist_to_obs(i+1) = dist;
end


% compare the ideal and actual trajectories
figure;
plot(xs(1,:), xs(2,:), '--g'); % plot final trajectory
hold on;
% plot final trajectory
plot(x_actual(1,start_index:end), x_actual(2,start_index:end), '-k'); 

% show axle circles at start and end
X = [x_actual(1,start_index); 
    x_actual(1,start_index) + S.l*cos(x_actual(3,start_index));
    x_actual(1, end);
    x_actual(1,end) + S.l*cos(x_actual(3,end))];
Y = [x_actual(2,start_index); 
    x_actual(2,start_index) + S.l*sin(x_actual(3,start_index));
    x_actual(2, end);
    x_actual(2,end) + S.l*sin(x_actual(3,end))];
R = S.circ_r*ones(4,1);
viscircles([X Y], R, 'Color', 'k', 'LineStyle', '--');

% just drawing circular obstacles
if isfield(S, 'os')
  da = .1;
  a = -da:da:2*pi;
  for i=1:length(S.os)
    % draw obstacle
    plot(S.os(i).p(1) + cos(a)*S.os(i).r,  S.os(i).p(2) + sin(a)*S.os(i).r, ...
         '-r','LineWidth',2);
  end
end

% plot exclusion zone boundary
if isfield(S, 'ez')
    for i=1:length(S.ez)
        plot([xs(1,1), xs(1,end)+S.l],[ -[S.ez(i).coeffs(1), ...
            S.ez(i).coeffs(3)]*[xs(1,1);1], -[S.ez(i).coeffs(1), ...
            S.ez(i).coeffs(3)]*[xs(1,end)+S.l;1] ], '--r');
    end
end

xlabel('x')
ylabel('y')
legend('Ideal Trajectory', 'Actual Trajectory', 'Location', 'northwest');
title('Ideal vs. Actual Trajectory')
axis equal

% compare the ideal and actual controls
figure;
plot(0:S.h:tf-S.h, us(1,:), '--b');
hold on;
plot(0:S.h:tf-S.h, us(2,:), '--r');
t_vec = 0:S.h:tf-S.h;
plot(t_vec(start_index:end), u_actual(1,start_index:end), '-b');
plot(t_vec(start_index:end), u_actual(2,start_index:end), '-r');
legend('u_{1,d}', 'u_{2,d}', 'u_{1,a}', 'u_{2,a}');
xlabel('t')
ylabel('u')
title('Ideal vs. Actual Controls')

% plot tracking error over time
figure;
plot(t_vec(start_index:end), tracking_error);
xlabel('t')
ylabel('e')
title('Tracking Error over Time')

% plot nearest distance to obstacle over time
figure;
plot(t_vec(start_index:end), dist_to_obs);
xlabel('t')
ylabel('|d_{obs}|')
title('Distance to Nearest Obstacle over Time')

end


% feedback control law derived via backstepping
function u = car_ctrl(x, S, i)
%%%%%%%%%%% compute nominal (noiseless) control (psi in notes) %%%%%%%%%%%
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
     v_ref^2*cos(theta_ref)/S.l, sin(theta_ref)]*[tan(u_ref(1)); u_ref(2)];

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

% augmented inputs u_aug =(tan(u1), u2)
R = [-v^2*sin(theta)/S.l, cos(theta);
     v^2*cos(theta)/S.l, sin(theta)];
u_aug = inv(R)*(-k2*z - e + d2yd - k1*e_dot);

%%%%%%%%%%% computing disturbance rejection term v %%%%%%%%%%%

w1 = v/S.l * ( (k1*(x(1)-x_ref(1)) - v_ref*cos(theta_ref) + ...
    v*cos(theta))*(-v*sin(theta)) + (k1*(x(2)-x_ref(2)) - ...
    v_ref*sin(theta_ref) + v*sin(theta))*(v*cos(theta))  );

w2 = (k1*(x(1)-x_ref(1)) - v_ref*cos(theta_ref) + ...
    v*cos(theta))*(cos(theta)) + (k1*(x(2)-x_ref(2)) - ...
    v_ref*sin(theta_ref) + v*sin(theta))*(sin(theta));

w = [w1; w2];

% k_eta must be at least k_noise, could be greater
k_eta = 1*S.k_tot; 
eta = k_eta*abs(v);
% piecewise form of u_v to prevent chattering
eps = 1e-4;
if eta*norm(w) >= eps
    u_v = (-eta/norm(w)) * w;
else
    u_v = -(eta)^2/eps * w;
end


u_aug = u_aug + u_v; % add disturbance rejection term

% convert from u_aug to u
u = [atan(u_aug(1)); u_aug(2)];

% restrict controls to limits
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

if (k == S.N+1) % if you're at the end of the trajectory, get terminal cost
  if isfield(S, 'xd') % if a desired final state is specified, use error
        xfError = x - S.xd; 
  else
        xfError = x; % else, the origin is the implied final desired state
  end
  L = xfError'*S.Qf*xfError/2; % standard quadratic error
  Lx = S.Qf*xfError;
  Lxx = S.Qf;
  Lu = [];
  Luu = [];
else
  L = S.h/2*(x'*S.Q*x + u'*S.R*u); % else, accumulate running cost
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

% quadratic penalty term for exclusion zone violations
if isfield(S, 'ez')
  for i=1:length(S.ez) % for each obstacle
    for j = 0:1 % for each collision checking circle (on each axle)
        circ_center = [x(1) + j*S.l*cos(x(3)); x(2) + j*S.l*sin(x(3))];
              
        satisfied = true;
        
        if S.ez(i).sign == ">=" % saying car must be above this line
            theta_tmp = atan2(-S.ez(i).coeffs(1), S.ez(i).coeffs(2));
            alpha_tmp = pi/2 - theta_tmp;
            delta_y = circ_center(2) - (-S.ez(i).coeffs(1)*circ_center(1) - ...
                S.ez(i).coeffs(3));
            dist = delta_y*sin(alpha_tmp);
            c = dist - S.circ_r;
            % need a g vec for error computation (from boundary to circ)
            g = dist*S.ez(i).coeffs(1:2); % normal dist * gradient of boundary
            
            if c < 0 % not enough clearance
                satisfied = false;
            end
        else % saying car must be below this line
            
            theta_tmp = atan2(-S.ez(i).coeffs(1), S.ez(i).coeffs(2));
            alpha_tmp = pi/2 - theta_tmp;
            delta_y = (-S.ez(i).coeffs(1)*circ_center(1) - ...
                S.ez(i).coeffs(3)) - circ_center(2);
            dist = delta_y*sin(alpha_tmp);
            c = dist - S.circ_r;
            % need a g vec for error computation (from boundary to circ)
            g = -dist*S.ez(i).coeffs(1:2); % -normal dist * gradient of boundary
            
            if c < 0 % not enough clearance
                satisfied = false;
            end
            
        end
        
        if satisfied == true
         continue
        end
        
        % converting back to positive value
        c = -c;
    
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