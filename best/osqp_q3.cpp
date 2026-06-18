#include <iostream>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <osqp-eigen/osqp-eigen.h>

int main() {
    // 定义QP
    // 决策变量: x = [x, y]^T
    // 目标函数: 0.5 * x^T * P * x + q^T * x
    // P = [1, 0; 0, 10], q = [-3, -30]^T
    // 约束: x + y <= 4  =>  [1, 1] * x <= 4

    Eigen::SparseMatrix<double> P(2, 2);
    P.insert(0, 0) = 1.0;
    P.insert(1, 1) = 10.0;//这个是矩阵P

    Eigen::VectorXd q(2);
    q << -3.0, -30.0;//线性向量q

    Eigen::SparseMatrix<double> A(1, 2);
    A.insert(0, 0) = 1.0;
    A.insert(0, 1) = 1.0;// 约束矩阵A


    Eigen::VectorXd lowerBound(1);
    lowerBound << -1e9;  // 用一个大负数代表 -inf
    // 约束下界 l 和上界 u: l <= A*x <= u
    // 这里只有上界约束，下界为 -inf

    Eigen::VectorXd upperBound(1);
    upperBound << 4.0;//约束的上界

    //配置并实例化求解器
    OsqpEigen::Solver solver;

    // 设置求解器参数
    solver.settings()->setVerbosity(false);// 关闭调试输出
    solver.settings()->setWarmStart(true);// 启用热启动

    // 将问题数据填入求解器
    solver.data()->setNumberOfVariables(2);//决策变量中变量个数，这里是x和y
    solver.data()->setNumberOfConstraints(1);//约束条件个数
    solver.data()->setHessianMatrix(P);//二次方的系数矩阵
    solver.data()->setGradient(q);//一次方的系数矩阵
    solver.data()->setLinearConstraintsMatrix(A);//约束矩阵A
    solver.data()->setLowerBound(lowerBound);//下界
    solver.data()->setUpperBound(upperBound);//上界u=4

    if (!solver.initSolver()) {
        std::cerr << "求解器初始化失败！" << std::endl;
        return 1;
    }

    if (!solver.solve()) {
        std::cerr << "问题求解失败！" << std::endl;
        return 1;
    }

    //输出结果
    Eigen::VectorXd solution = solver.getSolution();

    std::cout << "最优解 x: " << solution(0) << std::endl;
    std::cout << "最优解 y: " << solution(1) << std::endl;
    std::cout << "约束 x+y = " << solution(0) + solution(1) << " (应 <= 4)" << std::endl;

    return 0;
}