%% Clear previous work
if exist('pycom')
    pycom.close()
end
clc;
clear all;
close all;

%% Define parameters
Ts = 0.125;   % Model descretizazion step
H = 0.5;     % Horizon

% Robot pole lengths
l1 = [0.377; 0.074];
l2 = [0.461; -0.104];
l3 = [0.272; 0.0];

% Lidar specs
lidar_N = 3;
lidar_Nt = 150;
lidar_ang = pi*3/4;

%% Load python communication wrapper for env
[a,b,c] = pyversion;
if c ~= 1
    pyversion '/usr/bin/python3';
end
% Add python path to envs
flag = int32(bitor(2, 8));
py.sys.setdlopenflags(flag);

if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end
pycom = py.importlib.import_module('communicator');
py.importlib.reload(pycom);
pycom.setup();


%% Set up acado problem
BEGIN_ACADO;
    % Define initial state as input variable
    x0 = acado.MexInputVector;
    
    % System inputs
    Control u_x u_y u_th u_j1 u_j2;
    
    %% Define Differential Equations
    % Define differential states of setpoint, joints and base
    DifferentialState sp_x sp_y j1_p j2_p j1_v j2_v mb_v_x mb_v_y mb_v_th;
    DifferentialState l1_x l1_y l2_x l2_y l3_x l3_y;
    Control j1_p_eps j2_p_eps j1_v_eps j2_v_eps;
    Control mb_v_x_eps mb_v_y_eps mb_v_th_eps;
    % Define differential states of lidar scans
    for li=1:lidar_N
         DifferentialState(sprintf('lx_%i', li));
         DifferentialState(sprintf('ly_%i', li));
         Control(sprintf('leps_%i', li));
         Control(sprintf('lepsl1_%i', li));
         Control(sprintf('lepsl2_%i', li));
         Control(sprintf('lepsl3_%i', li));
    end
    

    % Tools
    R = @(a) [cos(a), -sin(a); sin(a) cos(a)];
    C = @(a) [0, -a; a, 0];

    % Equations
    f_sp = is(-(R(j1_p + j2_p)'*[mb_v_x; mb_v_y] ...
             + C(mb_v_th+j1_v)*R(j2_p)'*l1 ...
             + C(mb_v_th+j1_v+j2_v)*(l2+l3)));


    f = acado.DifferentialEquation();
    % Base velocities
    f.add(dot(mb_v_x) == u_x);
    f.add(dot(mb_v_y) == u_y);
    f.add(dot(mb_v_th) == u_th);
    % Arm positions
    f.add(dot(j1_p) == j1_v);
    f.add(dot(j2_p) == j2_v);
    % Arm velocities
    f.add(dot(j1_v) == u_j1);
    f.add(dot(j2_v) == u_j2);
    % Setpoint position
    f.add(dot(sp_x) == f_sp(1));
    f.add(dot(sp_y) == f_sp(2));
    % Link positions
    f_l1 = is(C(j1_v)*R(j1_p)*l1);
    f_l2 = is(f_l1 + C(j2_v)*R(j1_p+j2_p)*l2);
    f_l3 = is(f_l1 + C(j2_v)*R(j1_p+j2_p)*(l2+l3));
    f.add(dot(l1_x) == f_l1(1));
    f.add(dot(l1_y) == f_l1(2));
    f.add(dot(l2_x) == f_l1(1));
    f.add(dot(l2_y) == f_l1(2));
    f.add(dot(l3_x) == f_l1(1));
    f.add(dot(l3_y) == f_l1(2));
    % Lidar
    for li=1:lidar_N
        x = eval(sprintf('lx_%i', li));
        y = eval(sprintf('ly_%i', li));
        f_li = is([-mb_v_x; -mb_v_y] + C(-mb_v_th)*[x;y]);
        f.add(dot(x) == f_li(1));
        f.add(dot(y) == f_li(2));
    end
    


    %% Create optimal control problem 
    % Obtain variable lengths
    n_X = length(diffStates);   % # states
    n_U = length(controls);     % # inputs

    % Set up solvers
    acadoSet('problemname', 'nmpc');
    ocp = acado.OCP(0.0, H, round(H/Ts));
    
    % Set up cost function
    ocp.minimizeLSQ(sp_x);
    ocp.minimizeLSQ(sp_y);
    
    % Set dynamics and limits
    ocp.subjectTo( f );
    ocp.subjectTo(-1.45 <= u_x <= 1.45);
    ocp.subjectTo(-1.45 <= u_y <= 1.45);
    ocp.subjectTo(-0.45 <= u_th <= 0.45);
    ocp.subjectTo(-0.7 <= u_j1 <= 0.7);
    ocp.subjectTo(-0.7 <= u_j2 <= 0.7);
    
    % Joint 1
    ocp.subjectTo(-2.89 <= j1_p + j1_p_eps);
    ocp.subjectTo(j1_p - j1_p_eps <= 2.89);
    ocp.subjectTo(j1_p_eps >= 0);
    ocp.minimizeLSQ(j1_p_eps*100);
    
    ocp.subjectTo(-1.0 <= j1_v + j1_v_eps);
    ocp.subjectTo(j1_v - j1_v_eps <= 1.0);
    ocp.subjectTo(j1_v_eps >= 0);
    ocp.minimizeLSQ(j1_v_eps*100);
    
    % Joint 2
    ocp.subjectTo(0.05 <= j2_p + j2_p_eps);
    ocp.subjectTo(j2_p - j2_p_eps <= 3.0);
    ocp.subjectTo(j2_p_eps >= 0);
    ocp.minimizeLSQ(j2_p_eps*100);
    
    ocp.subjectTo(-21.0 <= j2_v + j2_v_eps);
    ocp.subjectTo(j2_v - j2_v_eps <= 1.0);
    ocp.subjectTo(j2_v_eps >= 0);
    ocp.minimizeLSQ(j2_v_eps*100);
    
    % Base
    ocp.subjectTo(-0.35 <= mb_v_x + mb_v_x_eps);
    ocp.subjectTo(mb_v_x - mb_v_x_eps <= 0.35);
    ocp.subjectTo(mb_v_x_eps >= 0);
    ocp.minimizeLSQ(mb_v_x_eps*100);
    
    ocp.subjectTo(-0.35 <= mb_v_y + mb_v_y_eps);
    ocp.subjectTo(mb_v_y - mb_v_y_eps <= 0.35);
    ocp.subjectTo(mb_v_y_eps >= 0);
    ocp.minimizeLSQ(mb_v_y_eps*100);
    
    ocp.subjectTo(-0.7 <= mb_v_th + mb_v_th_eps);
    ocp.subjectTo(mb_v_th - mb_v_th_eps <= 0.7);
    ocp.subjectTo(mb_v_th_eps >= 0);
    ocp.minimizeLSQ(mb_v_th_eps*100);
    
    % Lidar
    for li=1:lidar_N
        x = eval(sprintf('lx_%i', li));
        y = eval(sprintf('ly_%i', li));
        eps = eval(sprintf('leps_%i', li));
        epsl1 = eval(sprintf('lepsl1_%i', li));
        epsl2 = eval(sprintf('lepsl2_%i', li));
        epsl3 = eval(sprintf('lepsl3_%i', li));
        
        ocp.subjectTo( 0.35 <= x^2 + y^2 + eps );
        ocp.subjectTo( eps >= 0 );
        ocp.minimizeLSQ(eps*100);
        ocp.subjectTo( 0.15 <= (x-l1_x)^2 + (y-l1_y)^2 + epsl1 );
        ocp.subjectTo( epsl1 >= 0 );
        ocp.minimizeLSQ(epsl1*100);
        ocp.subjectTo( 0.15 <= (x-l2_x)^2 + (y-l2_y)^2 + epsl2 );
        ocp.subjectTo( epsl2 >= 0 );
        ocp.minimizeLSQ(epsl2*100);
        ocp.subjectTo( 0.05 <= (x-l3_x)^2 + (y-l3_y)^2 + epsl3 );
        ocp.subjectTo( epsl3 >= 0 );
        ocp.minimizeLSQ(epsl3*100);
        
%         ocp.subjectTo(x^2 + y^2 >= 0.5);
%         ocp.subjectTo((x-l1_x)^2 + (y-l1_y)^2 >= 0.2);
%         ocp.subjectTo((x-l2_x)^2 + (y-l2_y)^2 >= 0.1);
%         ocp.subjectTo((x-l3_x)^2 + (y-l3_y)^2 >= 0.05);
    end

    % Create real time algorithm and specify params
    algo = acado.RealTimeAlgorithm(ocp, 0.02);
    algo.set('KKT_TOLERANCE', 1e-2);
    algo.set('INTEGRATOR_TOLERANCE', 1e-3);
%     algo.set('MAX_NUM_ITERATIONS', 3 );
%     algo.set('HESSIAN_APPROXIMATION', 'GAUSS_NEWTON' );
%     algo.set('DISCRETIZATION_TYPE', 'MULTIPLE_SHOOTING');
%     algo.set( 'MAX_NUM_QP_ITERATIONS', 500 );
%     algo.set( 'HOTSTART_QP', 'YES' );
%     algo.set( 'LEVENBERG_MARQUARDT', 1e-10 );
    % algo.set( 'NUM_INTEGRATOR_STEPS', round(H/Ts)); % does not exist
    % algo.set( 'QP_SOLVER', 'QP_QPOASES' ); % does not exist
    % algo.set( 'INTEGRATOR_TYPE', 'INT_IRK_GL4' ); % Hangs up matlab
    % algo..set( 'SPARSE_QP_SOLUTION', 'FULL_CONDENSING' ); % not impl.
    
    % Create controller and add initial constraint
    controller = acado.Controller(algo);
    controller.init(0, x0);
    controller.step(0, x0);

END_ACADO;


%% Run controller
% Obtain initial observation
obs_r = cellfun(@double,cell(pycom.get_obs()));
for i=1:1000
    % Process raw observation
    x0 = process_obs(obs_r, lidar_N, lidar_Nt, lidar_ang, l1, l2, l3);
    % Calculate optimal solution
    out = nmpc_RUN(x0);
    u = process_inp(out.U)
    
    % Apply to simulation and obtain new observation
    obs_r = cellfun(@double,cell(pycom.step(py.list(u))));
end

%% Define input / output processing functions
function obs_p = process_obs(obs_r, lidar_N, lidar_Nt, lidar_ang, l1, l2, l3)
    R = @(a) [cos(a), -sin(a); sin(a) cos(a)];
    C = @(a) [0, -a; a, 0];
    
    obs_r
    obs_p = zeros(1, 9+3*2+lidar_N*2);
    obs_p(1:9) = [obs_r(3), -obs_r(2), ...
                  obs_r(46), -obs_r(47), obs_r(48), -obs_r(49), ...
                  obs_r(7), obs_r(8), obs_r(9)];

    % Calculate link poses
    obs_p(10:11) = R(obs_p(3))*l1;
    obs_p(12:13) = R(obs_p(3))*(l1 + R(obs_p(4))*l2);
    obs_p(14:15) = R(obs_p(3))*(l1 + R(obs_p(4))*(l2+l3));
    % Calculate lidar poses (x, y)
    sc = obs_r(50:50+lidar_Nt-1);
   % sc(23:27) = 5;
    sc2 = zeros(1, lidar_Nt);
   
    th = linspace(-lidar_ang+pi/2, lidar_ang+pi/2, lidar_Nt);
    %polarscatter(th, sc)
    %pause(0.01)
    if lidar_Nt ~= lidar_N
        sc_f = min(reshape(sc, lidar_Nt/lidar_N, []));
    else
        sc_f = sc;
    end
    ff = lidar_N/lidar_Nt*lidar_ang;
    angs = linspace(-ff, ff, lidar_N);
    sc_x = cos(angs).*sc_f;
    sc_y = sin(angs).*sc_f;
    
    angs = linspace(-lidar_ang, lidar_ang, lidar_Nt);
    sc_xt = cos(angs).*sc;
    sc_yt = sin(angs).*sc;
    points = [sc_xt' sc_yt'];
    
    obs_p(16:end) = reshape(cat(1, sc_x, sc_y), 1, []);
    
    
    
    plot(points(:,1),points(:,2),'o');
    hold on


    [x1 y1 x2 y2] = get_lines(points);

    plot(x1, y1, 'g-')
    plot(x2, y2, 'g-')
    legend('Noisy points','Least squares fit','Robust fit');
    hold off
    pause(0.01);
end



function inp_p = process_inp(inp_r)
    inp_p = inp_r(1:5);
    inp_p(5) = -inp_p(5);
end

function [x1 y1 x2 y2] = get_lines(points)
    sampleSize = 2; % number of points to sample per trial
    maxDistance = 0.2; % max allowable distance for inliers

    fitLineFcn = @(points) polyfit(points(:,1),points(:,2),1); % fit function using polyfit
    evalLineFcn = ...   % distance evaluation function
      @(model, points) sum((points(:, 2) - polyval(model, points(:,1))).^2,2);

    [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
      sampleSize,maxDistance);
    modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
    inlierPts = points(inlierIdx,:);
    x1 = [min(inlierPts(:,1)) max(inlierPts(:,1))];
    y1 = modelInliers(1)*x1 + modelInliers(2);
    points = points(~inlierIdx,:);
    
    [modelRANSAC, inlierIdx] = ransac(points,fitLineFcn,evalLineFcn, ...
      sampleSize,maxDistance);
    modelInliers = polyfit(points(inlierIdx,1),points(inlierIdx,2),1);
    inlierPts = points(inlierIdx,:);
    x2 = [min(inlierPts(:,1)) max(inlierPts(:,1))];
    y2 = modelInliers(1)*x2 + modelInliers(2);
end