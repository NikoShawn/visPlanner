#include <ros/ros.h>
#include <quadrotor_msgs/FovFaces.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Point.h>
#include <map>
#include <vector>
#include <string>

// 全局变量存储 FovFaces 数据
std::map<std::string, std::vector<geometry_msgs::Point>> faces;

// 解析 FovFaces 消息数据，返回面的顶点字典
std::map<std::string, std::vector<geometry_msgs::Point>> parse_faces_data(const quadrotor_msgs::FovFaces::ConstPtr& data) {
    std::map<std::string, std::vector<geometry_msgs::Point>> parsed_faces;

    // 解析 face1
    for (const auto& p : data->face1) {
        parsed_faces["face1"].push_back(p);
    }

    // 解析 face2
    for (const auto& p : data->face2) {
        parsed_faces["face2"].push_back(p);
    }

    // 解析 face3
    for (const auto& p : data->face3) {
        parsed_faces["face3"].push_back(p);
    }

    // 解析 face4
    for (const auto& p : data->face4) {
        parsed_faces["face4"].push_back(p);
    }

    // 解析 face5
    for (const auto& p : data->face5) {
        parsed_faces["face5"].push_back(p);
    }

    return parsed_faces;
}

// FOV 话题回调函数，解析数据并更新全局变量 `faces`
void fovFacesCallback(const quadrotor_msgs::FovFaces::ConstPtr& data) {
    faces = parse_faces_data(data);  // 解析数据
    // ROS_INFO("FovFaces data has been updated!");
}

// 示例函数：使用全局变量中的 FovFaces 数据
void process_faces_data() {
    for (const auto& face_pair : faces) {
        const std::string& face_name = face_pair.first;
        const std::vector<geometry_msgs::Point>& face_points = face_pair.second;

        ROS_INFO("Processing %s with %zu points...", face_name.c_str(), face_points.size());

        // 示例：计算每个面的几何中心
        geometry_msgs::Point center;
        center.x = 0.0;
        center.y = 0.0;
        center.z = 0.0;

        for (const auto& point : face_points) {
            center.x += point.x;
            center.y += point.y;
            center.z += point.z;
        }

        if (!face_points.empty()) {
            center.x /= face_points.size();
            center.y /= face_points.size();
            center.z /= face_points.size();
            ROS_INFO("  Center of %s: (%f, %f, %f)", face_name.c_str(), center.x, center.y, center.z);
        }
    }
}

// 回调函数，当接收到ESDFMap消息时会被调用
void esdfCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // 打印接收到的Float32MultiArray消息的基本信息
    // ROS_INFO("Received ESDFMap message!");

    // // 打印数组的布局信息
    // ROS_INFO("Array layout:");
    // ROS_INFO("  Dimensions: %zu", msg->layout.dim.size());  // 使用 %zu 格式化 size_t
    // for (size_t i = 0; i < msg->layout.dim.size(); ++i) {
    //     ROS_INFO("    Dimension %zu: label = %s, size = %d, stride = %d",
    //              i, msg->layout.dim[i].label.c_str(), msg->layout.dim[i].size, msg->layout.dim[i].stride);
    // }

    // // 打印数组的前10个值
    // ROS_INFO("First 10 ESDF values:");
    // for (size_t i = 0; i < 10 && i < msg->data.size(); ++i) {
    //     ROS_INFO("  ESDF[%zu] = %f", i, msg->data[i]);  // 使用 %zu 格式化 size_t
    // }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "my_node");
    ros::NodeHandle nh;

    // 订阅FovFaces话题
    ros::Subscriber fov_sub = nh.subscribe("/drone_0_traj_server/fov_faces", 10, fovFacesCallback);

    // 订阅ESDFMap话题
    ros::Subscriber esdf_sub = nh.subscribe("/drone_0_planning/esdf_data", 10, esdfCallback);

    // ROS_INFO("Hello, ROS! Waiting for messages...");
    ros::spin();
    return 0;
}