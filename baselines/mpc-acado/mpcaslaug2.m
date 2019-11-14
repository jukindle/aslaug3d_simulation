%% Clear previous work
if exist('pycom')
    try
    pycom.close()
    end
end
clc;
clear all;
close all;
%% Define parameters
params = struct;
params.MPC.Ts = 0.3;   % Model descretizazion step
params.MPC.H = 3.0;     % Horizon

% Robot pole lengths
% params.arm.l1 = [0.377; 0.074];
% params.arm.l2 = [0.461; -0.104];
% params.arm.l3 = [0.272; 0.0];
params.arm.l1 = [0.316; 0.082];
params.arm.l2 = [0.384; -0.083];
params.arm.l3 = [0.211; 0.0];

% Lidar specs
params.lidar.Nt = 150;
params.lidar.ang = pi*3/4;

% Safety distances
params.constraints.r_base = 0.6;
params.constraints.r_l1 = 0.25;
params.constraints.r_l2 = 0.15;
params.constraints.r_l3 = 0.001;


%% Set up acado problem
BEGIN_ACADO;        
    %% Define states and inputs of OCP
    % Define: Initial state
    x0 = acado.MexInputVector;
    
    % Define: Control inputs
    Control u_x u_y u_th u_j1 u_j2;
    
    % Define: System states
    DifferentialState sp_x sp_y j1_p j2_p j1_v j2_v mb_v_x mb_v_y mb_v_th;
    DifferentialState l1_x l1_y l2_x l2_y l3_x l3_y;
    DifferentialState w1x w1y w2x w2y;
    
    % Define: Slack variables
    Control j_p_eps j_v_eps mb_v_eps bw_eps lw_eps;
    
    % Define: Helper functions
    R = @(a) [cos(a), -sin(a); sin(a) cos(a)];
    C = @(a) [0, -a; a, 0];

    %% Set up system dynamics
    f = acado.DifferentialEquation();
         
    % Dynamics: Base velocities
    f.add(dot(mb_v_x) == u_x);
    f.add(dot(mb_v_y) == u_y);
    f.add(dot(mb_v_th) == u_th);
    
    % Dynamics: Arm positions
    f.add(dot(j1_p) == j1_v);
    f.add(dot(j2_p) == j2_v);
    
    % Dynamics: Arm velocities
    f.add(dot(j1_v) == u_j1);
    f.add(dot(j2_v) == u_j2);
    
    % Dynamics: Setpoint position
    f_sp = is(-(R(j1_p + j2_p)'*[mb_v_x; mb_v_y] ...
             + C(mb_v_th+j1_v)*R(j2_p)'*params.arm.l1 ...
             + C(mb_v_th+j1_v+j2_v)*(params.arm.l2+params.arm.l3)));
    f.add(dot(sp_x) == f_sp(1));
    f.add(dot(sp_y) == f_sp(2));
    
    % Dynamics: Link positions
    f_l1 = is(C(j1_v)*R(j1_p)*params.arm.l1);
    f_l2 = is(f_l1 + C(j2_v)*R(j1_p+j2_p)*params.arm.l2);
    f_l3 = is(f_l1 + C(j2_v)*R(j1_p+j2_p)*(params.arm.l2+params.arm.l3));
    f.add(dot(l1_x) == f_l1(1));
    f.add(dot(l1_y) == f_l1(2));
    f.add(dot(l2_x) == f_l2(1));
    f.add(dot(l2_y) == f_l2(2));
    f.add(dot(l3_x) == f_l3(1));
    f.add(dot(l3_y) == f_l3(2));
    
    % Dynamics: Wall positions
    f_lid1 = is(-C(mb_v_th)*[w1x; w1y] - [mb_v_x; mb_v_y]);
    f_lid2 = is(-C(mb_v_th)*[w2x; w2y] - [mb_v_x; mb_v_y]);
    f.add(dot(w1x) == f_lid1(1));
    f.add(dot(w1y) == f_lid1(2));
    f.add(dot(w2x) == f_lid2(1));
    f.add(dot(w2y) == f_lid2(2));

    %% Create optimal control problem 
    % Obtain variable lengths
    n_X = length(diffStates);
    n_U = length(controls);

    % Set up solvers
    acadoSet('problemname', 'nmpc');
    ocp = acado.OCP(0.0, params.MPC.H, round(params.MPC.H/params.MPC.Ts));
    
    %% Set up OCP constraints
    % Constraints: dynamics
    ocp.subjectTo( f );
    
    % Constraints: Input limits
    ocp.subjectTo(-1.45 <= u_x <= 1.45);
    ocp.subjectTo(-1.45 <= u_y <= 1.45);
    ocp.subjectTo(-0.45 <= u_th <= 0.45);
    ocp.subjectTo(-0.7 <= u_j1 <= 0.7);
    ocp.subjectTo(-0.7 <= u_j2 <= 0.7);
    
    % Constraints: Joint 1
    ocp.subjectTo(-2.89 <= j1_p + j_p_eps);
    ocp.subjectTo(j1_p - j_p_eps <= 2.89);
    
    ocp.subjectTo(-1.0 <= j1_v + j_v_eps);
    ocp.subjectTo(j1_v - j_v_eps <= 1.0);
    
    % Constraints: Joint 2
    ocp.subjectTo(0.05 <= j2_p + j_p_eps);
    ocp.subjectTo(j2_p - j_p_eps <= 2.5);
    
    ocp.subjectTo(-21.0 <= j2_v + j_v_eps);
    ocp.subjectTo(j2_v - j_v_eps <= 1.0);
    
    % Constraints: Base
    ocp.subjectTo(-0.35 <= mb_v_x + mb_v_eps);
    ocp.subjectTo(mb_v_x - mb_v_eps <= 0.35);
    
    ocp.subjectTo(-0.35 <= mb_v_y + mb_v_eps);
    ocp.subjectTo(mb_v_y - mb_v_eps <= 0.35);
    
    ocp.subjectTo(-0.7 <= mb_v_th + mb_v_eps);
    ocp.subjectTo(mb_v_th - mb_v_eps <= 0.7);
    
    
    % Constraints: Wall-base distance
    w1r = is((w1x+0.3)^2 + w1y^2);
    ocp.subjectTo(params.constraints.r_base^2 <= w1r + bw_eps);
    w2r = is((w2x+0.3)^2 + w2y^2);
    ocp.subjectTo(params.constraints.r_base^2 <= w2r + bw_eps);
    
    % Constraints: Wall-link1 distance
    l1w1s = is((w1y*l1_x-w1x*l1_y)/(w1x^2 + w1y^2));
    l1w1x = is(w1x + l1w1s*w1y);
    l1w1y = is(w1y - l1w1s*w1x);
    l1w1e = is((l1w1x-l1_x)^2+(l1w1y-l1_y)^2);
    l1w2s = is((w2y*l1_x-w2x*l1_y)/(w2x^2 + w2y^2));
    l1w2x = is(w2x + l1w2s*w2y);
    l1w2y = is(w2y - l1w2s*w2x);
    l1w2e = is((l1w2x-l1_x)^2+(l1w2y-l1_y)^2);
    ocp.subjectTo(params.constraints.r_l1^2 <= l1w1e + lw_eps);
    ocp.subjectTo(params.constraints.r_l1^2 <= l1w2e + lw_eps);
    
    % Constraints: Wall-link2 distance
    l2w1s = is((w1y*l2_x-w1x*l2_y)/(w1x^2 + w1y^2));
    l2w1x = is(w1x + l2w1s*w1y);
    l2w1y = is(w1y - l2w1s*w1x);
    l2w1e = is((l2w1x-l2_x)^2+(l2w1y-l2_y)^2);
    l2w2s = is((w2y*l2_x-w2x*l2_y)/(w2x^2 + w2y^2));
    l2w2x = is(w2x + l2w2s*w2y);
    l2w2y = is(w2y - l2w2s*w2x);
    l2w2e = is((l2w2x-l2_x)^2+(l2w2y-l2_y)^2);
    ocp.subjectTo(params.constraints.r_l2^2 <= l2w1e + lw_eps);
    ocp.subjectTo(params.constraints.r_l2^2 <= l2w2e + lw_eps);
    
    % Constraints: Wall-link3 distance
    l3w1s = is((w1y*l3_x-w1x*l3_y)/(w1x^2 + w1y^2));
    l3w1x = is(w1x + l3w1s*w1y);
    l3w1y = is(w1y - l3w1s*w1x);
    l3w1e = is((l3w1x-l3_x)^2+(l3w1y-l3_y)^2);
    l3w2s = is((w2y*l3_x-w2x*l3_y)/(w2x^2 + w2y^2));
    l3w2x = is(w2x + l3w2s*w2y);
    l3w2y = is(w2y - l3w2s*w2x);
    l3w2e = is((l3w2x-l3_x)^2+(l3w2y-l3_y)^2);
    ocp.subjectTo(params.constraints.r_l3^2 <= l3w1e + lw_eps);
    ocp.subjectTo(params.constraints.r_l3^2 <= l3w2e + lw_eps);
    
    % Constraints: Slack variables
    ocp.subjectTo(mb_v_eps >= 0);
    ocp.subjectTo(j_p_eps >= 0);
    ocp.subjectTo(j_v_eps >= 0);
    ocp.subjectTo(bw_eps >= 0);
    ocp.subjectTo(lw_eps >= 0);
    
    %% Set up cost function
    % Cost: Setpoint
    ocp.minimizeLSQEndTerm((sp_x)*10000, 0);
    ocp.minimizeLSQEndTerm((sp_y)*10000, 0);
    
    % Cost: Base velocity
    ocp.minimizeLSQEndTerm(mb_v_x*10, 0);
    ocp.minimizeLSQEndTerm(mb_v_y*10, 0);
    
    % Cost: Joint velocity
    ocp.minimizeLSQEndTerm(j1_v*5, 0);
    ocp.minimizeLSQEndTerm(j2_v*5, 0);
 
    % Cost: Soft constraints
    ocp.minimizeLSQ(mb_v_eps*10);
    ocp.minimizeLSQ(j_p_eps*1e3);
    ocp.minimizeLSQ(j_v_eps*10);
    ocp.minimizeLSQ(bw_eps*1e3);
    ocp.minimizeLSQ(lw_eps*1e3);
    
    
    %% Create MPC algorithm
    algo = acado.RealTimeAlgorithm(ocp, 0.2);
%     algo.set('KKT_TOLERANCE', 1e-2);
%     algo.set('INTEGRATOR_TOLERANCE', 1e-3);
%     algo.set('MAX_NUM_ITERATIONS', 3 );
%     algo.set('HESSIAN_APPROXIMATION', 'GAUSS_NEWTON' );
%     algo.set('DISCRETIZATION_TYPE', 'MULTIPLE_SHOOTING');
%     algo.set( 'MAX_NUM_QP_ITERATIONS', 500 );
%     algo.set( 'HOTSTART_QP', 'YES' );
%     algo.set( 'LEVENBERG_MARQUARDT', 1e-10 );
%     algo.set( 'NUM_INTEGRATOR_STEPS', round(H/params.MPC.Ts)); % does not exist
%     algo.set( 'QP_SOLVER', 'QP_QPOASES' ); % does not exist
%     algo.set( 'INTEGRATOR_TYPE', 'INT_IRK_GL4' ); % Hangs up matlab
%     algo.set( 'SPARSE_QP_SOLUTION', 'FULL_CONDENSING' ); % not impl.
    
    % Create controller and add initial constraint
    controller = acado.Controller(algo);
    controller.init(0, x0);
    controller.step(0, x0);

END_ACADO;


%% Load python communication wrapper for env
[a,b,c] = pyversion;
if c ~= 1
    pyversion '/usr/bin/python3';
end
% Add python path to envs
flag = int32(bitor(2, 8));
py.sys.setdlopenflags(flag);

if count(py.sys.path, '') == 0
    insert(py.sys.path,int32(0),'');
end
pycom = py.importlib.import_module('communicator');
try
py.importlib.reload(pycom);
end
pycom.setup();

%% Run controller
% Obtain initial observation
obs_r = cellfun(@double,cell(pycom.get_obs()));
figure();
% pause(15);

dt_list = ones(1, 20);
for i=1:1000
    % Process raw observation
    x0 = process_obs(obs_r, params);
    % Calculate optimal solution
    ts = cputime;
    out = nmpc_RUN(x0);
    te = cputime;
    dt_list = [ te-ts dt_list(1:end-1)];
    disp(['FPS: ', num2str(size(dt_list, 2)/sum(dt_list))]);
    u = process_inp(out.U)
    disp('Slack vars');
    disp('[j_p     j_v     mb_v    bw      lw]')
    disp(out.U(6:end));
    
    % Apply to simulation and obtain new observation
    for j=1:8
        obs_r = cellfun(@double,cell(pycom.step(py.list(u))));
    end
end

%% Define input / output processing functions
function obs_p = process_obs(obs_r, params)
    R = @(a) [cos(a), -sin(a); sin(a) cos(a)];
    C = @(a) [0, -a; a, 0];
    
    obs_p = zeros(1, 9+3*2 + 4);
    obs_p(1:9) = [obs_r(3), -obs_r(2), ...
                  obs_r(46), -obs_r(47), obs_r(48), -obs_r(49), ...
                  obs_r(7), obs_r(8), obs_r(9)];

    % Calculate link poses
    obs_p(10:11) = R(obs_p(3))*params.arm.l1;
    obs_p(12:13) = R(obs_p(3))*(params.arm.l1 + R(obs_p(4))*params.arm.l2);
    obs_p(14:15) = R(obs_p(3))*(params.arm.l1 + R(obs_p(4))*(params.arm.l2+params.arm.l3));
    
    % Calculate lidar poses (x, y)
    sc = obs_r(50:50+params.lidar.Nt-1);
    angs = linspace(-params.lidar.ang, params.lidar.ang, params.lidar.Nt);
    sc_xt = cos(angs).*sc;
    sc_yt = sin(angs).*sc;
    if sum(sc <= 4.95) == 0
        sc = sc -0.1;
    end
    points = [sc_xt(sc <= 4.95)' sc_yt(sc <= 4.95)'];
    points = points - [0.093 0];
    

    [x1 y1 x2 y2] = get_lines(points);
    % Nearest point on both lines
    [nx1 ny1] = get_nearest_point_on_line(x1, y1);
    [nx2 ny2] = get_nearest_point_on_line(x2, y2);
    
    obs_p(16:end) = [nx1 ny1 ...
                     nx2 ny2];
                 
    % Project setpoint on to wall
    sp_rf = R(obs_p(3)) * (params.arm.l1 + R(obs_p(4))*(params.arm.l2+params.arm.l3+obs_p(1:2)'));
    [px1 py1] = get_nearest_point_on_line(x1-sp_rf(1), y1-sp_rf(2));
    [px2 py2] = get_nearest_point_on_line(x2-sp_rf(1), y2-sp_rf(2));
    if (px1^2 + py1^2) <= (px2^2 + py2^2)
        rn = (R(obs_p(3)+obs_p(4))'*[px1; py1])';
    else
        rn = (R(obs_p(3)+obs_p(4))'*[px2; py2])';
    end
    % TODO: fix me
    shift = 0.05;
    res_p = obs_p(1:2)+rn + rn/norm(rn)*shift;
    res_m = obs_p(1:2)+rn - rn/norm(rn)*shift;
    if norm(res_p) <= norm(res_m)
        obs_p(1:2) = res_p;
    else
        obs_p(1:2) = res_m;
    end
    
    %% Plotting
    plot([0], [0], '^', 'LineWidth', 5);
    hold on;
    % Plot lidar scan
    plot(points(:,1),points(:,2),'o');
    % Plot detected walls
    plot(x1, y1, 'g-')
    plot(x2, y2, 'g-')
    % Plot nearest points of walls
    plot([nx1 nx2], [ny1 ny2], '+', 'LineWidth', 5);
    % Plot robot arm
    plot([0 obs_p(10) obs_p(12) obs_p(14)], [0 obs_p(11) obs_p(13) obs_p(15)], 'LineWidth', 2);
    % Plot setpoint
    sp_rf = R(obs_p(3)) * (params.arm.l1 + R(obs_p(4))*(params.arm.l2+params.arm.l3+obs_p(1:2)'));
    plot(sp_rf(1), sp_rf(2), 'x', 'LineWidth', 7);
    % Plot robot contour
    plot([0.166 0.166 -0.766 -0.766 0.166], [0.397 -0.397 -0.397 0.397 0.397]);
    % Plot safety distances
    viscircles([-0.3 0], params.constraints.r_base, 'LineWidth', 1);
    viscircles([obs_p(10) obs_p(11)], params.constraints.r_l1, 'LineWidth', 1);
    viscircles([obs_p(12) obs_p(13)], params.constraints.r_l2, 'LineWidth', 1);
    viscircles([obs_p(14) obs_p(15)], params.constraints.r_l3, 'LineWidth', 1);
    % Set figure scaling and show
    axis([-5 5 -5 5]);
    hold off
    pause(0.001);
end
function inp_p = process_inp(inp_r)
    inp_p = inp_r(1:5);
    inp_p(5) = -inp_p(5);
end

function [nx ny] = get_nearest_point_on_line(x, y)
    R = @(a) [cos(a), -sin(a); sin(a) cos(a)];
    C = @(a) [0, -a; a, 0];
    
    
    px = x(1);
    py = y(1);
    lx = x(2) - x(1);
    ly = y(2) - y(1);
    
    ny = lx * (lx*py - ly*px)/(lx^2 + ly^2);
    nx = -ny*ly/lx;
   
    
end
function [x1 y1 x2 y2] = get_lines(points)
    sampleSize = 2; % number of points to sample per trial
    maxDistance = 0.05; % max allowable distance for inliers

    fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1); % fit function using polyfit
    evalLineFcn = ...   % distance evaluation function
      @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2,2);

    [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
      sampleSize,maxDistance, 'Confidence', 90);
    modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
    inlierPts = points(inlierIdx,:);
%     x1 = [min(inlierPts(:,1)) max(inlierPts(:,1))];
    x1 = [-10 10];
    y1 = polyval(modelInliers, x1);
%     y1 = modelInliers(1)*x1 + modelInliers(2);
    points = points(~inlierIdx,:);
    
    diss = abs((y1(2)-y1(1))*points(:, 1) - (x1(2)-x1(1))*points(:, 2) + x1(2)*y1(1) - y1(2)*x1(1))/sqrt((y1(2)-y1(1))^2+(x1(2)-x1(1))^2);
    points = points(diss >= 1.0, :);
    
    if size(points, 1) < 2
        x2 = x1
        y2 = y1
        return
    end
    
    fitLineFcn = @(points) [modelRANSAC(1) points(1, 2) - modelRANSAC(1)*points(1, 1)]; % fit function using polyfit

    [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
      1,maxDistance, 'Confidence', 99);
    %modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
    inlierPts = points(inlierIdx,:);
%     x2 = [min(inlierPts(:,1)) max(inlierPts(:,1))];
    x2 = [-10 10];
    y2 = polyval(modelRANSAC, x2);
%     y2 = modelInliers(1)*x2 + modelInliers(2);
end
function closeSimulation()
    if exist('pycom')
        try
            pycom.close()
        end
    end
end