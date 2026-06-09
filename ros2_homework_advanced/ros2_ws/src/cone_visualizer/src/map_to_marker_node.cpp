#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "fsd_common_msgs/msg/map.hpp"

using namespace std::chrono_literals;

class MarkerPublisher : public rclcpp::Node
{
public:
    MarkerPublisher() : Node("marker_publisher")
    {
        marker_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "visualization_marker", 10);

        map_sub_ = this->create_subscription<fsd_common_msgs::msg::Map>(
            "/estimation/slam/map", 10,
            std::bind(&MarkerPublisher::mapCallback, this, std::placeholders::_1));
    }

private:

    void mapCallback(const fsd_common_msgs::msg::Map::SharedPtr msg)
    {
        int id = 0;

        // 黄色锥桶
        for (auto &cone : msg->cone_yellow) {
            publishCone(cone.position.x, cone.position.y, cone.position.z,
                        id++, 1.0, 1.0, 0.0, msg->header.frame_id);
        }

        // 蓝色锥桶
        for (auto &cone : msg->cone_blue) {
            publishCone(cone.position.x, cone.position.y, cone.position.z,
                        id++, 0.0, 0.0, 1.0, msg->header.frame_id);
        }

        // 红色锥桶
        for (auto &cone : msg->cone_red) {
            publishCone(cone.position.x, cone.position.y, cone.position.z,
                        id++, 1.0, 0.0, 0.0, msg->header.frame_id);
        }

        // 未知锥桶
        for (auto &cone : msg->cone_unknown) {
            publishCone(cone.position.x, cone.position.y, cone.position.z,
                        id++, 0.5, 0.5, 0.5, msg->header.frame_id);
        }
    }

    void publishCone(float x, float y, float z,
                     int id,
                     float r, float g, float b,
                     const std::string &frame_id)
    {
        visualization_msgs::msg::Marker marker;

        marker.header.frame_id = frame_id;
        marker.header.stamp = this->now();

        marker.ns = "cones";
        marker.id = id;

        marker.type = visualization_msgs::msg::Marker::SPHERE;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;

        marker.pose.orientation.w = 1.0;

        marker.scale.x = 0.3;
        marker.scale.y = 0.3;
        marker.scale.z = 0.3;

        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 1.0;

        marker.lifetime = rclcpp::Duration::from_seconds(0.0);

        marker_pub_->publish(marker);
    }

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;
    rclcpp::Subscription<fsd_common_msgs::msg::Map>::SharedPtr map_sub_;
};

int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MarkerPublisher>());
    rclcpp::shutdown();
    return 0;
}