%% Q3: 机器狗约束优化 (MATLAB quadprog 求解)
clear; clc;

% 定义 QP 矩阵
H = [1 0; 0 10];
f = [-3; -30];

% 约束 (x + y <= 4)
A = [1 1];
b = 4;

% 变量边界
lb = [-inf; -inf];
ub = [inf; inf];

% 求解
x_opt = quadprog(H, f, A, b, [], [], lb, ub);

% 输出
fprintf('最优解 x: %.6f\n', x_opt(1));
fprintf('最优解 y: %.6f\n', x_opt(2));
fprintf('x + y = %.6f (应 <= 4)\n', x_opt(1)+x_opt(2));