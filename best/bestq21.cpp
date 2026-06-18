#include <iostream>      
#include <Eigen/Dense>   
#include <cmath>         // 用于计算范数
#include <fstream>  //用于数据输出

int main() {
    
    Eigen::Vector2d x(0.0,0.0);

    double target_x = 3.0;
    double target_y = 3.0;

    double eta = 0.1;//依旧典中典之学习率

    double ending = 1e-3;

    int max_iter = 10000;
    int iter = 0;

    // 新增：打开文件用于输出数据
    std::ofstream out("gradient_data.csv");
    out << "iter,x,y,cost" << std::endl;  

    while (iter < max_iter)
    {
        double grad_x = x(0)-target_x;
        double grad_y = 10.0*(x(1)-target_y);//推导写在pdf了，这里对应的是推导里的步骤

        double grad_norm = sqrt(pow(grad_x,2)+pow(grad_y,2));
        if(grad_norm < ending)
        {
            break;
        }
        x(0) = x(0)-eta*grad_x;
        x(1) = x(1)-eta*grad_y;

        iter+=1;

        //每步记录数据
        double cost = 0.5*pow(x(0)-3,2)+5.0*pow(x(1)-3,2);
        out << iter << "," << x(0) << "," << x(1) << "," << cost << std::endl;
    }
    double value_counting = 0.5*pow(x(0)-3,2)+5.0*pow(x(1)-3,2);//计算代价函数
    std::cout<< "迭代次数: " << iter << std::endl;
    std::cout<< "最优解x: (" << x(0) << "," << x(1) << ")" << std::endl;
    std::cout<< "代价函数值: " << value_counting << std::endl;    

    //由于是一个比较独立的环境，我想尝试一下利用,matplotlib实现可视化，问了ai发现有cpp版本的库，直接用！
    //超级意外情况..系统磁盘不够了，装不下这个库，所以我打算输出数据然后给py进行实现。


    // 关闭文件
    out.close();

    return 0;
}