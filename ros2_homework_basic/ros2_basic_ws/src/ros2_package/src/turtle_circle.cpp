#include<cmath>
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"

class TurtleCircle : public rclcpp::Node
{
public:
    TurtleCircle() : Node("turtle_circle")
    {
        // 发布器
        pub_ = this->create_publisher<geometry_msgs::msg::Twist>("turtle1/cmd_vel", 10);

        // 定时器
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
            std::bind(&TurtleCircle::timer_callback, this)
        );

        
        /*//设置速度
        linear_speed_ = 1.0;
        angular_speed_ = 1.0;
        switch_time_ = M_PI/angular_speed_;
        这个是最开始写一个文件的时候写的,为了方便初始参数的调教
        */
        /*
        linear_speed_ = this->declare_parameter("linear_speed",1.0);//线速度，也就是切向速度
        angular_speed_ = this->declare_parameter("angular_speed",1.0);//转动速度，每秒转动多少角度
        switch_time_ = (2*M_PI)/angular_speed_;
        这个是第二次，我发现曲线轨迹不是完全闭合，所以继续优化
        */
        linear_speed_ = this->declare_parameter("linear_speed", 1.0);
        angular_speed_ = this->declare_parameter("angular_speed", 1.0);
        direction_ = true;//旋转的方向
        last_time_ = this->now();//用于计算dt
        angle_accum_ = 0.0;//用来累加旋转的角度

        // 用 wall clock 初始化
        //rclcpp::Clock clock(RCL_ROS_TIME);  因为刚才编译完之后shell提示now和lastswitchtime不是同一个时间表，所以要统一
        //last_switch_time_ = clock.now();    这个是用来对齐时间的
    }

private:
//声明成员函数和成员变量   
void timer_callback()
{
    auto now = this->now();
    // 计算真实时间间隔
    dt_ = (now-last_time_).seconds();
    last_time_ = now;
    // 累积角度
    angle_accum_ += angular_speed_ * dt_;
    // 每2π切换一次方向（圆）
    if (angle_accum_ >= 2*M_PI)//这里是2π，经常搞错，之前的callback函数也搞错。其实前面也写了几个版本，不想太长于是我给删掉了
    {
        direction_ = !direction_;
        angle_accum_ = 0.0;//换方向并清零已经走过的角度
    }
    geometry_msgs::msg::Twist msg;
    msg.linear.x = linear_speed_;
    msg.angular.z = direction_ ? angular_speed_ : -angular_speed_;//条件表达式：条件？结果1：结果2；
    pub_->publish(msg);
}
    //成员变量与成员对象
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    double linear_speed_;
    double angular_speed_;
    double switch_time_;
    double dt_;
    double angle_accum_;

    rclcpp::Time last_time_;
    rclcpp::Time last_switch_time_;

    bool direction_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TurtleCircle>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
/*
很可惜的是，就算进行了两个版本的迭代，还是会出现8字失真的情况。。
第一次想着用时间来直接计算，然后发现偏差大的可怕，基本上跑完第一圈就会出现肉眼可见的偏差，而且及其快速地产生横向偏移
然后就试着用对运动轨迹求积分，求完了发现好像还是有问题，只是偏移没那么快了 不知道怎么优化了，可能要完美的话得用极坐标？
*/