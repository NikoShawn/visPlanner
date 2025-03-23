#include <ros/ros.h>
#include <quadrotor_msgs/FovFaces.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Point.h>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>

// 定义ESDF点的结构体
struct ESDFPoint {
    double x, y, z;  // 坐标
    double distance; // 距离值
    
    // 构造函数
    ESDFPoint(double _x, double _y, double _z, double _dist) 
        : x(_x), y(_y), z(_z), distance(_dist) {}
};

struct ObstacleRegion {
    int id;                                // 障碍物ID
    ESDFPoint min_angle_point;            // 最小角度点
    ESDFPoint max_angle_point;            // 最大角度点
    std::pair<double, double> min_intersection; // 最小角度射线与FOV边界的交点 (x, y)
    std::pair<double, double> max_intersection; // 最大角度射线与FOV边界的交点 (x, y)
    // 添加构造函数
    ObstacleRegion(int _id = 0, 
        const ESDFPoint& _min = ESDFPoint(0, 0, 0, 0), 
        const ESDFPoint& _max = ESDFPoint(0, 0, 0, 0))
    : id(_id), min_angle_point(_min), max_angle_point(_max),
    min_intersection(std::make_pair(0.0, 0.0)),
    max_intersection(std::make_pair(0.0, 0.0)) {}
};

// 全局变量
std::vector<ObstacleRegion> obstacle_regions;

// 存储ESDF数据的容器
std::vector<ESDFPoint> esdf_points;

// 存储裁剪后的ESDF数据（在FOV区域内的点）
std::vector<ESDFPoint> fov_esdf_points;

// 用于存储障碍物点的容器
std::vector<ESDFPoint> obstacles;

// 存储被遮挡区域中离本机最近的点
struct OccludedRegion {
    ESDFPoint closest_point;
    double distance_to_origin;
    
    OccludedRegion(const ESDFPoint& point, double dist) 
        : closest_point(point), distance_to_origin(dist) {}
};

// 全局变量存储被遮挡区域中离本机最近的点
std::vector<std::pair<ESDFPoint, int>> occluded_regions;

// 全局变量存储 FovFaces 数据
std::map<std::string, std::vector<geometry_msgs::Point>> faces;

struct FaceCoordinates {
    std::vector<std::pair<double, double>> vertices; // 5 个点的 (x, y) 坐标
};

// 全局变量存储 5 个点的坐标
FaceCoordinates face_coordinates;

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

// 判断点(x,y)是否在三角形内部
bool isPointInTriangle(double x, double y, 
                       const std::pair<double, double>& v1, 
                       const std::pair<double, double>& v2, 
                       const std::pair<double, double>& v3) {
    // 计算三角形的面积
    double area = 0.5 * std::abs((v2.first - v1.first) * (v3.second - v1.second) - 
                                 (v3.first - v1.first) * (v2.second - v1.second));
    
    // 计算点与三角形顶点构成的三个三角形的面积
    double area1 = 0.5 * std::abs((v1.first - x) * (v2.second - y) - 
                                  (v2.first - x) * (v1.second - y));
    double area2 = 0.5 * std::abs((v2.first - x) * (v3.second - y) - 
                                  (v3.first - x) * (v2.second - y));
    double area3 = 0.5 * std::abs((v3.first - x) * (v1.second - y) - 
                                  (v1.first - x) * (v3.second - y));
    
    // 如果三个小三角形的面积之和等于大三角形的面积，则点在三角形内部
    // 使用浮点数比较，需要考虑精度问题
    double sum = area1 + area2 + area3;
    return std::abs(sum - area) < 1e-9;
}

// 使用射线法检测被障碍物遮挡的区域，并找出每个区域中离本机最近的点
void detectOccludedRegions() {
    // 清空之前的结果
    occluded_regions.clear();
    
    if (face_coordinates.vertices.size() < 3 || fov_esdf_points.empty()) {
        ROS_WARN("FOV vertices or ESDF points not initialized yet");
        return;
    }
    
    // 如果没有检测到障碍物，直接返回
    if (obstacles.empty()) {
        ROS_INFO("没有检测到障碍物");
        return;
    }

    // 本机位置（FOV的第一个顶点）
    const auto& origin = face_coordinates.vertices[0];
    const auto& fov1 = face_coordinates.vertices[1];
    const auto& fov2 = face_coordinates.vertices[2];
    
    // 计算fov1和fov2连成的直线方程 ax + by + c = 0
    double a = fov2.second - fov1.second;  // y2 - y1
    double b = fov1.first - fov2.first;    // x1 - x2
    double c = fov2.first * fov1.second - fov1.first * fov2.second;  // x2*y1 - x1*y2

    // 计算每个障碍物点相对于本机的角度
    std::vector<std::pair<ESDFPoint, double>> obstacles_with_angles;
    for (const auto& obs : obstacles) {
        // 直接使用ESDFPoint的x和y成员
        // 计算障碍物相对于本机的向量
        double dx = obs.x - origin.first;  // 假设origin是std::pair<double, double>
        double dy = obs.y - origin.second;
        
        // 计算角度（弧度），使用atan2确保角度在[-π, π]范围内
        double angle = std::atan2(dy, dx);
        
        obstacles_with_angles.push_back({obs, angle});
        
        // 打印每个障碍物的角度（用于调试）
        // ROS_INFO("障碍物位置 (%.2f, %.2f), 角度: %.2f 弧度 (%.2f 度)", 
        //           obs.x, obs.y, angle, angle * 180.0 / M_PI);
    }

    // 按照角度大小对障碍物点进行排序
    std::sort(obstacles_with_angles.begin(), obstacles_with_angles.end(),
    [](const auto& j, const auto& k) {
        return j.second < k.second;
    });

    // 按照角度对障碍物点进行分类
    std::vector<std::vector<std::pair<ESDFPoint, double>>> classified_obstacles;

    // 角度阈值（可以根据需要调整）
    const double angle_threshold = 0.1; // 约5.7度
    
    if (!obstacles_with_angles.empty()) {
        // 创建第一个障碍物类别
        classified_obstacles.push_back({obstacles_with_angles[0]});
        
        // 遍历剩余的障碍物点
        for (size_t i = 1; i < obstacles_with_angles.size(); ++i) {
            const auto& current = obstacles_with_angles[i];
            const auto& previous = obstacles_with_angles[i-1];
            
            // 如果当前点与前一个点的角度差小于阈值，则归为同一类
            if (std::abs(current.second - previous.second) < angle_threshold) {
                classified_obstacles.back().push_back(current);
            } else {
                // 否则创建新的障碍物类别
                classified_obstacles.push_back({current});
            }
        }
    }

    // 提取每个障碍物中最大和最小角度的点
    std::vector<std::pair<std::pair<ESDFPoint, double>, std::pair<ESDFPoint, double>>> obstacle_boundaries;

    for (const auto& obstacle_group : classified_obstacles) {
        // 由于已经按角度排序，第一个点是最小角度，最后一个点是最大角度
        const auto& min_angle_point = obstacle_group.front();
        const auto& max_angle_point = obstacle_group.back();
        
        obstacle_boundaries.push_back({min_angle_point, max_angle_point});
    }

    // 清空之前的结果
    obstacle_regions.clear();

    // 计算每个障碍物边界点与原点组成的线和FOV边界直线的交点
    for (size_t i = 0; i < obstacle_boundaries.size(); ++i) {
        const auto& boundary = obstacle_boundaries[i];
        const auto& min_point = boundary.first.first;  // 最小角度点的ESDFPoint
        const auto& max_point = boundary.second.first; // 最大角度点的ESDFPoint
        
        // 创建新的障碍物区域
        ObstacleRegion region;
        region.id = i;
        region.min_angle_point = min_point;
        region.max_angle_point = max_point;
        
        // 计算从原点到最小角度点的方向向量
        double min_dx = min_point.x - origin.first;
        double min_dy = min_point.y - origin.second;
        
        // 计算从原点到最大角度点的方向向量
        double max_dx = max_point.x - origin.first;
        double max_dy = max_point.y - origin.second;
        
        // 计算最小角度射线的交点
        double min_t_denominator = a * min_dx + b * min_dy;
        
        if (std::abs(min_t_denominator) > 1e-6) {  // 避免除以零
            double min_t = -(a * origin.first + b * origin.second + c) / min_t_denominator;
            
            // 只考虑射线前方的交点 (t > 0)
            if (min_t > 0) {
                double min_intersection_x = origin.first + min_t * min_dx;
                double min_intersection_y = origin.second + min_t * min_dy;
                
                region.min_intersection = {min_intersection_x, min_intersection_y};
            } else {
                // 如果没有正向交点，可以设置一个默认值或标记
                region.min_intersection = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
            }
        } else {
            // 如果射线与FOV边界平行或重合，可以设置一个默认值或标记
            region.min_intersection = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        }
        
        // 计算最大角度射线的交点
        double max_t_denominator = a * max_dx + b * max_dy;
        
        if (std::abs(max_t_denominator) > 1e-6) {  // 避免除以零
            double max_t = -(a * origin.first + b * origin.second + c) / max_t_denominator;
            
            // 只考虑射线前方的交点 (t > 0)
            if (max_t > 0) {
                double max_intersection_x = origin.first + max_t * max_dx;
                double max_intersection_y = origin.second + max_t * max_dy;
                
                region.max_intersection = {max_intersection_x, max_intersection_y};
            } else {
                // 如果没有正向交点，可以设置一个默认值或标记
                region.max_intersection = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
            }
        } else {
            // 如果射线与FOV边界平行或重合，可以设置一个默认值或标记
            region.max_intersection = {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        }
        
        // 将障碍物区域添加到全局变量
        obstacle_regions.push_back(region);
        
        }
        // 调试用
        // ROS_INFO("共保存了 %zu 个障碍物区域", obstacle_regions.size());
        // for (size_t i = 0; i < obstacle_regions.size(); ++i) {
        //     const auto& region = obstacle_regions[i];
            
        //     ROS_INFO("障碍物 #%d 的坐标点:", region.id);
        //     ROS_INFO("  最小角度点: (%.2f, %.2f, %.2f), 距离值: %.2f", 
        //              region.min_angle_point.x, region.min_angle_point.y, 
        //              region.min_angle_point.z, region.min_angle_point.distance);
        //     ROS_INFO("  最大角度点: (%.2f, %.2f, %.2f), 距离值: %.2f", 
        //              region.max_angle_point.x, region.max_angle_point.y, 
        //              region.max_angle_point.z, region.max_angle_point.distance);
            
        //     // 如果交点有效，也输出交点坐标
        //     if (!std::isnan(region.min_intersection.first)) {
        //         ROS_INFO("  最小角度射线与FOV边界的交点: (%.2f, %.2f)", 
        //                  region.min_intersection.first, region.min_intersection.second);
        //     }
            
        //     if (!std::isnan(region.max_intersection.first)) {
        //         ROS_INFO("  最大角度射线与FOV边界的交点: (%.2f, %.2f)", 
        //                  region.max_intersection.first, region.max_intersection.second);
        //     }
            
        //     ROS_INFO("----------------------------");
        // }
}


// 裁剪ESDF点，只保留在FOV区域内的点
void clipEsdfPointsToFov() {
    // 确保FOV顶点已经初始化
    if (face_coordinates.vertices.size() < 3) {
        ROS_WARN("FOV vertices not initialized yet, cannot clip ESDF points");
        return;
    }
    
    // 清空之前裁剪的结果
    fov_esdf_points.clear();
    
    // 获取三角形的三个顶点
    const auto& v1 = face_coordinates.vertices[0];
    const auto& v2 = face_coordinates.vertices[1];
    const auto& v3 = face_coordinates.vertices[2];
    
    // 遍历所有ESDF点
    for (const auto& point : esdf_points) {
        // 只考虑x-y平面上的投影
        if (isPointInTriangle(point.x, point.y, v1, v2, v3)) {
            // 如果点在三角形内，则保存到裁剪后的容器中
            fov_esdf_points.push_back(point);
        }
    }
    
    // ROS_INFO("Clipped ESDF points: %zu out of %zu are within FOV", 
    //          fov_esdf_points.size(), esdf_points.size());
}

void detectObstacles(){
    obstacles.clear();
    // 遍历所有在FOV内的ESDF点
    for (const auto& point : fov_esdf_points) {
        // 检查ESDF值是否小于等于0（表示障碍物）
        if (point.distance <= 0.0) {
            // 将障碍物点添加到容器中
            obstacles.push_back(point);
        }
    }
    
    // 输出检测到的障碍物数量
    // ROS_INFO("detect %zu in fov", obstacles.size());
}

// FOV 话题回调函数，解析数据并更新全局变量 `faces`
void fovFacesCallback(const quadrotor_msgs::FovFaces::ConstPtr& data) {
    
    faces = parse_faces_data(data);  // 解析数据
    // 二维点调用
    face_coordinates.vertices.clear();
    face_coordinates.vertices.push_back(std::make_pair(faces["face1"][0].x, faces["face1"][0].y));
    for (size_t i = 0; i < 2; ++i) {
        face_coordinates.vertices.push_back(std::make_pair(faces["face5"][i].x, faces["face5"][i].y));
    }

    // std::cout << "face1: " <<face_coordinates.vertices[0].first << " " << face_coordinates.vertices[0].second << std::endl;
    // std::cout << "face5: " <<face_coordinates.vertices[1].first << " " << face_coordinates.vertices[1].second << std::endl;
    // std::cout << "face5: " <<face_coordinates.vertices[2].first << " " << face_coordinates.vertices[2].second << std::endl;
    // std::cout << "face5: " <<face_coordinates.vertices[3].first << " " << face_coordinates.vertices[3].second << std::endl;
    // std::cout << "face5: " <<face_coordinates.vertices[4].first << " " << face_coordinates.vertices[4].second << std::endl;

    // ROS_INFO("FovFaces data has been updated!");
}


// 回调函数，当接收到ESDFMap消息时会被调用
void esdfCallback(const std_msgs::Float32MultiArray::ConstPtr& msg) {
    // 清空之前的数据
    esdf_points.clear();
    
    // 遍历消息中的数据
    const std::vector<float>& data = msg->data;
    for (size_t i = 0; i < data.size(); i += 4) {
        if (i + 3 < data.size()) {
            double x = data[i];
            double y = data[i + 1];
            double z = data[i + 2];
            double distance = data[i + 3];
            
            // 将点和距离值存储到容器中
            esdf_points.emplace_back(x, y, z, distance);
        }
    }
    // 接收到新的ESDF数据后，立即进行裁剪
    // clipEsdfPointsToFov();
    // ROS_INFO("Received ESDF data with %zu points", esdf_points.size());
    // detectObstacles();
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "GMM_Planning_Test");
    ros::NodeHandle nh;

    // 订阅FovFaces话题
    ros::Subscriber fov_sub = nh.subscribe("/drone_0_traj_server/fov_faces", 10, fovFacesCallback);

    // 订阅ESDFMap话题
    ros::Subscriber esdf_sub = nh.subscribe("/drone_0_planning/esdf_data", 10, esdfCallback);
    
    // ROS_INFO("Hello, ROS! Waiting for messages...");
    // 创建一个定时器，定期检查和处理esdf_points
    ros::Rate rate(1); // 1Hz
    while (ros::ok()) {
        ros::spinOnce(); // 处理回调

        // 每次循环都尝试裁剪ESDF点
        if (!esdf_points.empty() && !face_coordinates.vertices.empty()) {
            clipEsdfPointsToFov();
            detectObstacles();
            detectOccludedRegions();

        rate.sleep();
        }
    }
}