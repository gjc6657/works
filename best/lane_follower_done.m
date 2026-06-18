%% 第二问：轨迹跟踪
clear; clc; close all

% 车辆参数
lfr = 2.168 + 1.907; % 轴距 L
dt = 0.01;
v = 15; 
sim_steps = 2000;

% 参考轨迹 (正弦曲线)
X_ref = 0:0.1:200; 
Y_ref = 10 * sin(X_ref / 15); 

% 初始车辆状态 
X = X_ref(1); Y = Y_ref(1) + 3; phi = 0; 
X_vec = zeros(1, sim_steps); Y_vec = zeros(1, sim_steps);


for ii = 1:sim_steps
    X_vec(ii) = X; Y_vec(ii) = Y;
    
    
    % ===============================================================
    
    % ================= TODO 2.1: 实现某种跟踪算法 =================
    %初始状态故意给了一个偏差+3，所以这里就要利用跟踪算法跟回去。
    %要做什么呢？找到目标点算夹角然后计算转向角

    dist = sqrt((X_ref - X).^2 + (Y_ref - Y).^2);
    [~, idx] = min(dist);
    Ld = 2;  % 前视距离，单位：米
    cum_dist = 0;
    target_x = X_ref(idx);
    target_y = Y_ref(idx);
    for j = idx : length(X_ref)-1
        seg_len = sqrt((X_ref(j+1)-X_ref(j))^2 + (Y_ref(j+1)-Y_ref(j))^2);
        if cum_dist + seg_len >= Ld
            ratio = (Ld - cum_dist) / seg_len;
            target_x = X_ref(j) + ratio * (X_ref(j+1)-X_ref(j));
            target_y = Y_ref(j) + ratio * (Y_ref(j+1)-Y_ref(j));
            break;
        end
        cum_dist = cum_dist + seg_len;
    end

    dx = target_x - X;
    dy = target_y - Y;
    alpha = atan2(dy, dx) - phi;
    sigma = atan2(2 * lfr * sin(alpha), Ld);

    % ===============================================================

    % ================= TODO 2.2: 车辆状态更新 =================
    % 提示: 将刚才求得的转向角 sigma 代入运动学模型（复用第一问代码），更新 X, Y, phi。
    
    phi_dot = v / lfr * tan(sigma);% 这个就是我推导过程中omega的部分
    phi = phi_dot*dt + phi% 实现航向角，也就是前轮角度的更新
    X = X + v * cos(phi) * dt% 实现X的参数更新
    Y = Y + v * sin(phi) * dt% 实现Y的参数更新
    % ===============================================================
    
    % 到达终点提前结束
    if X >= X_ref(end), break; end
end

% 绘图对比
figure; hold on; grid on;
plot(X_ref, Y_ref, 'k--', 'LineWidth', 2);
plot(X_vec(1:ii), Y_vec(1:ii), 'r-', 'LineWidth', 2);
legend('参考规划轨迹', '实际行驶轨迹');
title(['Pure Pursuit 跟踪 (Ld = ', num2str(Ld), 'm)']);
xlabel('X [m]'); ylabel('Y [m]'); axis equal;