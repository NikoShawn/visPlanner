#include <ros/ros.h>
#include <quadrotor_msgs/FovFaces.h>
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Point.h>
#include <map>
#include <vector>
#include <string>
#include <unordered_map>
#include <tuple>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

ros::Publisher nearest_points_pub;

// Global variables to store clustering results
std::vector<std::vector<size_t>> obstacle_clusters;
std::vector<size_t> obstacle_labels;

// Define 2D point structure for clustering
struct Point2D {
    double x;
    double y;
    
    // Calculate Euclidean distance between two points
    double distance(const Point2D& other) const {
        return std::sqrt(std::pow(x - other.x, 2) + std::pow(y - other.y, 2));
    }
};

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

struct Face3DCoordinates {
    std::vector<std::tuple<double, double, double>> vertices;
};

// Declare the variable
Face3DCoordinates face_coordinates_3d;

// 定义三维坐标结构体
std::map<std::string, std::vector<std::tuple<double, double, double>>> face_vertices_map;

// 定义全局结构体存储位置数据
struct DronePosition {
    double x;
    double y;
    double z;
    bool data_received;
    
    DronePosition() : x(0.0), y(0.0), z(0.0), data_received(false) {}
};

// 全局变量
DronePosition drone_position;

// 计算点到面的最近点的结构体
struct NearestPointInfo {
    std::string face_name;
    std::tuple<double, double, double> nearest_point;
    double distance;
};

// ----------------------------------Fov Faces----------------------------------

// 计算点到线段的最近点
std::tuple<double, double, double> nearestPointOnSegment(
    const std::tuple<double, double, double>& point,
    const std::tuple<double, double, double>& lineStart,
    const std::tuple<double, double, double>& lineEnd) {
    
    double x0, y0, z0, x1, y1, z1, x2, y2, z2;
    std::tie(x0, y0, z0) = point;
    std::tie(x1, y1, z1) = lineStart;
    std::tie(x2, y2, z2) = lineEnd;
    
    // 计算线段向量
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;
    
    // 计算线段长度的平方
    double lengthSquared = dx*dx + dy*dy + dz*dz;
    
    // 如果线段长度为0，则返回起点
    if (lengthSquared == 0.0) {
        return lineStart;
    }
    
    // 计算投影参数t
    double t = ((x0 - x1) * dx + (y0 - y1) * dy + (z0 - z1) * dz) / lengthSquared;
    
    // 限制t在[0,1]范围内
    t = std::max(0.0, std::min(1.0, t));
    
    // 计算最近点
    double nearestX = x1 + t * dx;
    double nearestY = y1 + t * dy;
    double nearestZ = z1 + t * dz;
    
    return std::make_tuple(nearestX, nearestY, nearestZ);
}

// 计算点到多边形的最近点
std::tuple<double, double, double> nearestPointOnPolygon(
    const std::tuple<double, double, double>& point,
    const std::vector<std::tuple<double, double, double>>& polygon) {
    
    if (polygon.empty()) {
        return point; // 如果多边形为空，返回原点
    }
    
    if (polygon.size() == 1) {
        return polygon[0]; // 如果多边形只有一个点，返回该点
    }
    
    // 初始化最近距离为无穷大
    double minDistanceSquared = std::numeric_limits<double>::max();
    std::tuple<double, double, double> nearestPoint;
    
    // 遍历多边形的所有边
    for (size_t i = 0; i < polygon.size(); ++i) {
        size_t j = (i + 1) % polygon.size(); // 下一个点的索引（循环回到起点）
        
        // 计算点到当前边的最近点
        auto pointOnSegment = nearestPointOnSegment(point, polygon[i], polygon[j]);
        
        // 计算距离的平方
        double x0, y0, z0, x1, y1, z1;
        std::tie(x0, y0, z0) = point;
        std::tie(x1, y1, z1) = pointOnSegment;
        double distanceSquared = (x1-x0)*(x1-x0) + (y1-y0)*(y1-y0) + (z1-z0)*(z1-z0);
        
        // 更新最近点
        if (distanceSquared < minDistanceSquared) {
            minDistanceSquared = distanceSquared;
            nearestPoint = pointOnSegment;
        }
    }
    
    return nearestPoint;
}

// Check if a point is inside a 3D polygon
bool isPointInPolygon(
    const std::tuple<double, double, double>& point,
    const std::vector<std::tuple<double, double, double>>& polygon,
    double nx, double ny, double nz) {
    
    if (polygon.size() < 3) {
        return false;
    }
    
    double px, py, pz;
    std::tie(px, py, pz) = point;
    
    // Find the best 2D projection plane based on the normal
    int dominant_axis = 0;
    double max_component = std::abs(nx);
    if (std::abs(ny) > max_component) {
        dominant_axis = 1;
        max_component = std::abs(ny);
    }
    if (std::abs(nz) > max_component) {
        dominant_axis = 2;
    }
    
    // Project to 2D by dropping one coordinate
    std::vector<std::pair<double, double>> polygon_2d;
    std::pair<double, double> point_2d;
    
    for (const auto& vertex : polygon) {
        double vx, vy, vz;
        std::tie(vx, vy, vz) = vertex;
        
        switch (dominant_axis) {
            case 0: // Drop x
                polygon_2d.push_back(std::make_pair(vy, vz));
                break;
            case 1: // Drop y
                polygon_2d.push_back(std::make_pair(vx, vz));
                break;
            case 2: // Drop z
                polygon_2d.push_back(std::make_pair(vx, vy));
                break;
        }
    }
    
    // Project the point too
    switch (dominant_axis) {
        case 0: // Drop x
            point_2d = std::make_pair(py, pz);
            break;
        case 1: // Drop y
            point_2d = std::make_pair(px, pz);
            break;
        case 2: // Drop z
            point_2d = std::make_pair(px, py);
            break;
    }
    
    // Use ray-casting algorithm in 2D
    bool inside = false;
    for (size_t i = 0, j = polygon_2d.size() - 1; i < polygon_2d.size(); j = i++) {
        if (((polygon_2d[i].second > point_2d.second) != (polygon_2d[j].second > point_2d.second)) &&
            (point_2d.first < (polygon_2d[j].first - polygon_2d[i].first) * (point_2d.second - polygon_2d[i].second) / 
             (polygon_2d[j].second - polygon_2d[i].second) + polygon_2d[i].first)) {
            inside = !inside;
        }
    }
    
    return inside;
}

// Calculate nearest points to faces, including projections onto face planes
std::vector<NearestPointInfo> calculateNearestPointsToFaces(
    const DronePosition& drone_position,
    const std::map<std::string, std::vector<std::tuple<double, double, double>>>& face_vertices_map) {
    
    std::vector<NearestPointInfo> result;
    
    // If drone position data hasn't been received, return empty result
    if (!drone_position.data_received) {
        return result;
    }
    
    // Drone position
    auto drone_point = std::make_tuple(drone_position.x, drone_position.y, drone_position.z);
    
    // Iterate through all faces
    for (const auto& face_entry : face_vertices_map) {
        const std::string& face_name = face_entry.first;
        const auto& face_vertices = face_entry.second;
        
        // Need at least 3 vertices to form a face
        if (face_vertices.size() < 3) {
            continue;
        }
        
        // Calculate face normal and a point on the face
        double x0, y0, z0, x1, y1, z1, x2, y2, z2;
        std::tie(x0, y0, z0) = face_vertices[0];
        std::tie(x1, y1, z1) = face_vertices[1];
        std::tie(x2, y2, z2) = face_vertices[2];
        
        // Compute two vectors on the plane
        double v1x = x1 - x0;
        double v1y = y1 - y0;
        double v1z = z1 - z0;
        
        double v2x = x2 - x0;
        double v2y = y2 - y0;
        double v2z = z2 - z0;
        
        // Compute face normal using cross product
        double nx = v1y * v2z - v1z * v2y;
        double ny = v1z * v2x - v1x * v2z;
        double nz = v1x * v2y - v1y * v2x;
        
        // Normalize the normal
        double norm = std::sqrt(nx*nx + ny*ny + nz*nz);
        if (norm < 1e-10) {
            // Degenerate face, skip it
            continue;
        }
        
        nx /= norm;
        ny /= norm;
        nz /= norm;
        
        // Calculate distance from drone to the face plane
        double dx = drone_position.x - x0;
        double dy = drone_position.y - y0;
        double dz = drone_position.z - z0;
        
        // Calculate signed distance to the plane
        double signed_distance = dx*nx + dy*ny + dz*nz;
        
        // Project the drone point onto the face plane
        double projected_x = drone_position.x - signed_distance * nx;
        double projected_y = drone_position.y - signed_distance * ny;
        double projected_z = drone_position.z - signed_distance * nz;
        
        auto projected_point = std::make_tuple(projected_x, projected_y, projected_z);
        
        // Check if the projected point is inside the polygon
        bool is_inside = isPointInPolygon(projected_point, face_vertices, nx, ny, nz);
        
        std::tuple<double, double, double> nearest_point;
        double distance;
        
        if (is_inside) {
            // If projected point is inside the face, it's the nearest point
            nearest_point = projected_point;
            distance = std::abs(signed_distance);
        } else {
            // If projected point is outside, find nearest point on boundary
            nearest_point = nearestPointOnPolygon(drone_point, face_vertices);
            
            double np_x, np_y, np_z;
            std::tie(np_x, np_y, np_z) = nearest_point;
            
            // Calculate Euclidean distance
            distance = std::sqrt(
                std::pow(np_x - drone_position.x, 2) + 
                std::pow(np_y - drone_position.y, 2) + 
                std::pow(np_z - drone_position.z, 2)
            );
        }
        
        // Create and add result
        NearestPointInfo info;
        info.face_name = face_name;
        info.nearest_point = nearest_point;
        info.distance = distance;
        
        result.push_back(info);
    }
    
    return result;
}


void printNearestPointsToFaces(const DronePosition& drone_position) {
    // Calculate nearest points
    auto nearest_points = calculateNearestPointsToFaces(drone_position, face_vertices_map);
    
    // Create a marker array to visualize the nearest points
    visualization_msgs::MarkerArray marker_array;
    
    for (size_t i = 0; i < nearest_points.size(); ++i) {
        const auto& info = nearest_points[i];
        double x, y, z;
        std::tie(x, y, z) = info.nearest_point;
        
        // Create a marker for each nearest point
        visualization_msgs::Marker marker;
        marker.header.frame_id = "world"; // Use your actual frame
        marker.header.stamp = ros::Time::now();
        marker.ns = "nearest_points";
        marker.id = i;
        marker.type = visualization_msgs::Marker::SPHERE;
        marker.action = visualization_msgs::Marker::ADD;
        
        // Set marker position
        marker.pose.position.x = x;
        marker.pose.position.y = y;
        marker.pose.position.z = z;
        marker.pose.orientation.w = 1.0;
        
        // Set marker scale
        marker.scale.x = 0.2;
        marker.scale.y = 0.2;
        marker.scale.z = 0.2;
        
        // Set marker color (different color for each face)
        float hue = static_cast<float>(i) / nearest_points.size();
        
        // Simple HSV to RGB conversion for distinct colors
        float r, g, b;
        if (hue < 1.0/3.0) {
            r = 1.0;
            g = 3.0 * hue;
            b = 0.0;
        } else if (hue < 2.0/3.0) {
            r = 1.0 - 3.0 * (hue - 1.0/3.0);
            g = 1.0;
            b = 3.0 * (hue - 1.0/3.0);
        } else {
            r = 0.0;
            g = 1.0 - 3.0 * (hue - 2.0/3.0);
            b = 1.0;
        }
        
        marker.color.r = r;
        marker.color.g = g;
        marker.color.b = b;
        marker.color.a = 0.8; // Slightly transparent
        
        // Lifetime of marker
        marker.lifetime = ros::Duration(1.0);
        
        // Add text marker to display the face name and distance
        visualization_msgs::Marker text_marker;
        text_marker.header = marker.header;
        text_marker.ns = "nearest_points_text";
        text_marker.id = i;
        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::Marker::ADD;
        
        text_marker.pose.position.x = x;
        text_marker.pose.position.y = y;
        text_marker.pose.position.z = z + 0.3; // Position text above the point
        text_marker.pose.orientation.w = 1.0;
        
        // Format text with face name and distance
        std::stringstream ss;
        ss << info.face_name << ": " << std::fixed << std::setprecision(2) << info.distance << "m";
        text_marker.text = ss.str();
        
        text_marker.scale.z = 0.15; // Text size
        text_marker.color.r = 1.0;
        text_marker.color.g = 1.0;
        text_marker.color.b = 1.0;
        text_marker.color.a = 0.8;
        text_marker.lifetime = ros::Duration(1.0);
        
        // Add markers to array
        marker_array.markers.push_back(marker);
        marker_array.markers.push_back(text_marker);
    }
    
    // Publish markers if we have at least one point and the publisher is valid
    if (!marker_array.markers.empty() && nearest_points_pub) {
        nearest_points_pub.publish(marker_array);
    }
}


// ----------------------------------Fov Faces----------------------------------

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
        ROS_INFO("no obstacles detected");
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



// DBSCAN algorithm for obstacle clustering
std::vector<std::vector<size_t>> dbscanObstacles(
    const std::vector<ESDFPoint>& obstacles, 
    double eps,        // Neighborhood radius
    int minPts         // Minimum points to form a cluster
) {
    const size_t n = obstacles.size();
    
    // Initialize visited status for all points
    std::vector<bool> visited(n, false);
    
    // Result: clusters of point indices
    std::vector<std::vector<size_t>> clusters;
    
    // Process each point
    for (size_t i = 0; i < n; i++) {
        // Skip if already visited
        if (visited[i]) continue;
        
        // Mark as visited
        visited[i] = true;
        
        // Find neighbors within eps distance
        std::vector<size_t> neighbors;
        for (size_t j = 0; j < n; j++) {
            if (i == j) continue;  // Skip self
            
            // Calculate distance between points (only x,y coordinates)
            double dist = std::sqrt(
                std::pow(obstacles[i].x - obstacles[j].x, 2) + 
                std::pow(obstacles[i].y - obstacles[j].y, 2)
            );
            
            if (dist <= eps) {
                neighbors.push_back(j);
            }
        }
        
        // Check if this is a core point
        if (neighbors.size() < static_cast<size_t>(minPts)) {
            // Not a core point, mark as noise
            continue;
        }
        
        // Start a new cluster
        std::vector<size_t> cluster;
        cluster.push_back(i);  // Add current point to cluster
        
        // Process all neighbors
        for (size_t j = 0; j < neighbors.size(); j++) {
            size_t neighborIdx = neighbors[j];
            
            // If not visited, mark as visited and check its neighborhood
            if (!visited[neighborIdx]) {
                visited[neighborIdx] = true;
                
                // Find neighbors of this neighbor
                std::vector<size_t> neighborNeighbors;
                for (size_t k = 0; k < n; k++) {
                    if (neighborIdx == k) continue;
                    
                    double dist = std::sqrt(
                        std::pow(obstacles[neighborIdx].x - obstacles[k].x, 2) + 
                        std::pow(obstacles[neighborIdx].y - obstacles[k].y, 2)
                    );
                    
                    if (dist <= eps) {
                        neighborNeighbors.push_back(k);
                    }
                }
                
                // If this is a core point, add its neighbors to the processing list
                if (neighborNeighbors.size() >= static_cast<size_t>(minPts)) {
                    for (size_t k = 0; k < neighborNeighbors.size(); k++) {
                        neighbors.push_back(neighborNeighbors[k]);
                    }
                }
            }
            
            // Add to cluster if not already in a cluster
            bool inCluster = false;
            for (const auto& existingCluster : clusters) {
                for (size_t clusterId : existingCluster) {
                    if (clusterId == neighborIdx) {
                        inCluster = true;
                        break;
                    }
                }
                if (inCluster) break;
            }
            
            if (!inCluster) {
                cluster.push_back(neighborIdx);
            }
        }
        
        // Add the cluster to the results
        clusters.push_back(cluster);
    }
    
    return clusters;
}

// Label points according to their cluster, noise points are labeled as 0
std::vector<size_t> labelObstacles(
    const std::vector<std::vector<size_t>>& clusters, 
    size_t n
) {
    std::vector<size_t> flatClusters(n, 0);  // Initialize all as noise (0)
    
    for (size_t i = 0; i < clusters.size(); i++) {
        for (auto pointIdx : clusters[i]) {
            flatClusters[pointIdx] = i + 1;  // Cluster IDs start from 1
        }
    }
    
    return flatClusters;
}

// Function to perform DBSCAN clustering on obstacles and visualize the results
void clusterObstacles(const std::vector<ESDFPoint>& obstacles, double eps = 0.5, int minPts = 5) {
    // Run DBSCAN algorithm
    obstacle_clusters = dbscanObstacles(obstacles, eps, minPts);
    obstacle_labels = labelObstacles(obstacle_clusters, obstacles.size());
    
    // Log clustering results
    ROS_INFO("Found %zu obstacle clusters", obstacle_clusters.size());
    
    // // Print number of points in each cluster
    // for (size_t i = 0; i < clusters.size(); i++) {
    //     ROS_INFO("Cluster %zu contains %zu points", i+1, clusters[i].size());
    // }
    
    // Count noise points
    // size_t noiseCount = std::count(labels.begin(), labels.end(), 0);
    // ROS_INFO("Detected %zu noise points", noiseCount);
    
    // Visualize clusters in RViz
    // visualization_msgs::MarkerArray cluster_markers;
    // int marker_id = 0;
    
    // // Create a marker for each cluster with a distinct color
    // for (size_t i = 0; i < clusters.size(); ++i) {
    //     // Choose a distinctive color for this cluster
    //     float hue = static_cast<float>(i) / clusters.size();
    //     float r, g, b;
        
    //     // Simple HSV to RGB conversion
    //     if (hue < 1.0/3.0) {
    //         r = 1.0;
    //         g = 3.0 * hue;
    //         b = 0.0;
    //     } else if (hue < 2.0/3.0) {
    //         r = 1.0 - 3.0 * (hue - 1.0/3.0);
    //         g = 1.0;
    //         b = 3.0 * (hue - 1.0/3.0);
    //     } else {
    //         r = 0.0;
    //         g = 1.0 - 3.0 * (hue - 2.0/3.0);
    //         b = 1.0;
    //     }
        
    //     // Create a marker for points in this cluster
    //     visualization_msgs::Marker points_marker;
    //     points_marker.header.frame_id = "world"; // Use appropriate frame
    //     points_marker.header.stamp = ros::Time::now();
    //     points_marker.ns = "obstacle_clusters";
    //     points_marker.id = marker_id++;
    //     points_marker.type = visualization_msgs::Marker::POINTS;
    //     points_marker.action = visualization_msgs::Marker::ADD;
    //     points_marker.pose.orientation.w = 1.0;
    //     points_marker.scale.x = 0.1;
    //     points_marker.scale.y = 0.1;
    //     points_marker.color.r = r;
    //     points_marker.color.g = g;
    //     points_marker.color.b = b;
    //     points_marker.color.a = 1.0;
    //     points_marker.lifetime = ros::Duration(1.0);
        
    //     // Add all points in this cluster
    //     for (auto point_idx : clusters[i]) {
    //         geometry_msgs::Point p;
    //         p.x = obstacles[point_idx].x;
    //         p.y = obstacles[point_idx].y;
    //         p.z = obstacles[point_idx].z;
    //         points_marker.points.push_back(p);
    //     }
        
    //     cluster_markers.markers.push_back(points_marker);
        
    //     // Create a text marker to label this cluster
    //     visualization_msgs::Marker text_marker;
    //     text_marker.header = points_marker.header;
    //     text_marker.ns = "obstacle_cluster_labels";
    //     text_marker.id = marker_id++;
    //     text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    //     text_marker.action = visualization_msgs::Marker::ADD;
        
    //     // Calculate centroid of the cluster for text placement
    //     double cx = 0, cy = 0, cz = 0;
    //     for (auto point_idx : clusters[i]) {
    //         cx += obstacles[point_idx].x;
    //         cy += obstacles[point_idx].y;
    //         cz += obstacles[point_idx].z;
    //     }
    //     cx /= clusters[i].size();
    //     cy /= clusters[i].size();
    //     cz /= clusters[i].size();
        
    //     text_marker.pose.position.x = cx;
    //     text_marker.pose.position.y = cy;
    //     text_marker.pose.position.z = cz + 0.2; // Slightly above the cluster
    //     text_marker.pose.orientation.w = 1.0;
        
    //     std::stringstream ss;
    //     ss << "Cluster " << (i + 1) << " (" << clusters[i].size() << " points)";
    //     text_marker.text = ss.str();
        
    //     text_marker.scale.z = 0.15; // Text size
    //     text_marker.color.r = 1.0;
    //     text_marker.color.g = 1.0;
    //     text_marker.color.b = 1.0;
    //     text_marker.color.a = 1.0;
    //     text_marker.lifetime = ros::Duration(1.0);
        
    //     cluster_markers.markers.push_back(text_marker);
    // }
    
    // // Create a marker for noise points (if any)
    // std::vector<size_t> noise_points;
    // for (size_t i = 0; i < obstacles.size(); ++i) {
    //     if (labels[i] == 0) {
    //         noise_points.push_back(i);
    //     }
    // }
    
    // if (!noise_points.empty()) {
    //     visualization_msgs::Marker noise_marker;
    //     noise_marker.header.frame_id = "world";
    //     noise_marker.header.stamp = ros::Time::now();
    //     noise_marker.ns = "obstacle_clusters";
    //     noise_marker.id = marker_id++;
    //     noise_marker.type = visualization_msgs::Marker::POINTS;
    //     noise_marker.action = visualization_msgs::Marker::ADD;
    //     noise_marker.pose.orientation.w = 1.0;
    //     noise_marker.scale.x = 0.05;
    //     noise_marker.scale.y = 0.05;
    //     noise_marker.color.r = 0.5;
    //     noise_marker.color.g = 0.5;
    //     noise_marker.color.b = 0.5;
    //     noise_marker.color.a = 0.5;
    //     noise_marker.lifetime = ros::Duration(1.0);
        
    //     for (auto point_idx : noise_points) {
    //         geometry_msgs::Point p;
    //         p.x = obstacles[point_idx].x;
    //         p.y = obstacles[point_idx].y;
    //         p.z = obstacles[point_idx].z;
    //         noise_marker.points.push_back(p);
    //     }
        
    //     cluster_markers.markers.push_back(noise_marker);
    // }
    
    // // Publish the marker array
    // static ros::Publisher cluster_pub = 
    //     ros::NodeHandle().advertise<visualization_msgs::MarkerArray>("/obstacle_clusters", 1);
    
    // if (!cluster_markers.markers.empty()) {
    //     cluster_pub.publish(cluster_markers);
    // }
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
    
    // If obstacles are detected, run clustering
    if (!obstacles.empty()) {
        // Run DBSCAN clustering with custom parameters
        // eps = 0.5 (radius in meters), minPts = 5 (minimum points per cluster)
        clusterObstacles(obstacles, 0.5, 5);
    }

    // 输出检测到的障碍物数量
    // ROS_INFO("detect %zu in fov", obstacles.size());
}

/**
 * 计算基于KL散度的权重
 * @param mu 均值向量集合，每列是一个均值向量
 * @param P_full 协方差矩阵集合
 * @param initial_mu 初始分布的均值向量
 * @param initial_P 初始分布的协方差矩阵
 * @return 计算得到的权重向量
 */

Eigen::VectorXd calculateKLBasedWeights(
    const Eigen::MatrixXd& mu,
    const std::vector<Eigen::MatrixXd>& P_full,
    const Eigen::VectorXd& initial_mu,
    const Eigen::MatrixXd& initial_P) {
    
    int n = mu.cols();  // 组件数量
    Eigen::VectorXd beta = Eigen::VectorXd::Ones(n);  // 默认权重为1
    
    if (n > 0) {
        // 计算每个组件到初始分布的KL散度
        int d = mu.rows();  // 维度
        Eigen::VectorXd kl_divergences = Eigen::VectorXd::Zero(n);
        
        // 预先计算初始分布的逆矩阵和行列式
        Eigen::MatrixXd initial_P_inv = initial_P.inverse();
        double initial_P_det = initial_P.determinant();
        
        // 计算每个组件的KL散度
        for (int i = 0; i < n; ++i) {
            // 计算均值差异
            Eigen::VectorXd diff = mu.col(i) - initial_mu;
            
            // 计算组件i到初始分布的KL散度
            kl_divergences(i) = 0.5 * (
                std::log(initial_P_det / P_full[i].determinant()) - d +
                (initial_P_inv * P_full[i]).trace() +
                diff.transpose() * initial_P_inv * diff
            );
        }
        
        // 计算权重为KL散度的倒数
        double eps = 1e-10;  // 避免除以零
        for (int i = 0; i < n; ++i) {
            beta(i) = 1.0 / (kl_divergences(i) + eps);
        }
        
        // 归一化权重使其和为1
        beta = beta / beta.sum();
    }
    
    return beta;
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

    // 清空之前的面顶点映射
    face_vertices_map.clear();
    
    // 三维点调用
    face_coordinates_3d.vertices.clear();

    // 遍历所有面并保存所有顶点到映射中
    for (const auto& face_pair : faces) {
        const std::string& face_name = face_pair.first;
        const std::vector<geometry_msgs::Point>& face_points = face_pair.second;
        
        // 为当前面创建顶点列表
        std::vector<std::tuple<double, double, double>> face_vertices;
        
        // 遍历当前面的所有顶点
        for (const auto& point : face_points) {
            // 添加到当前面的顶点列表
            face_vertices.push_back(std::make_tuple(point.x, point.y, point.z));
            
            // 同时添加到全局顶点列表
            face_coordinates_3d.vertices.push_back(std::make_tuple(point.x, point.y, point.z));
        }
        
        // 将面名称和对应的顶点存储到映射中
        face_vertices_map[face_name] = face_vertices;
    }

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

void odomCallback(const nav_msgs::Odometry::ConstPtr& msg){
    // 获取本机位置
    drone_position.x = msg->pose.pose.position.x;
    drone_position.y = msg->pose.pose.position.y;
    drone_position.z = msg->pose.pose.position.z;
    
    // 标记已接收数据
    drone_position.data_received = true;
    
    // ROS_INFO("本机位置: (%.2f, %.2f, %.2f)", drone_position.x, drone_position.y, drone_position.z);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "GMM_Planning_Test");
    ros::NodeHandle nh;

    // Initialize publishers
    nearest_points_pub = nh.advertise<visualization_msgs::MarkerArray>("/nearest_face_points", 1);


    // 订阅FovFaces话题
    ros::Subscriber fov_sub = nh.subscribe("/drone_0_traj_server/fov_faces", 10, fovFacesCallback);

    // 订阅ESDFMap话题
    ros::Subscriber esdf_sub = nh.subscribe("/drone_0_planning/esdf_data", 10, esdfCallback);

    // 订阅本机位置话题
    ros::Subscriber target_odom_sub = nh.subscribe("/drone_1_visual_slam/odom", 10, odomCallback);
    
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

            printNearestPointsToFaces(drone_position);
            
        rate.sleep();
        }
    }
}