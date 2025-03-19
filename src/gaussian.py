# -*- coding: utf-8 -*-
import rospy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib.animation import FuncAnimation
from quadrotor_msgs.msg import FovFaces  # 替换为实际的消息类型
from nav_msgs.msg import Odometry
import numpy as np
from tf.transformations import euler_from_quaternion
from scipy.stats import multivariate_normal

# 全局变量存储面和图形对象
faces = {}
polys = []
target_position = None
target_orientation = None
target_marker = None
intersection_markers = []
gaussian_plots = []
fig = None
ax = None

# 定义线条颜色
line_color = 'black'
intersection_color = 'red'  # 交点的颜色
gaussian_color = 'green'    # 高斯分布颜色

# 高斯分布参数
gaussian_std = 0.5  # 标准差，控制高斯分布的扩散程度
gaussian_points = 20  # 用于绘制高斯分布的点数量

def fov_callback(data):
    """
    FOV 话题回调函数，解析数据并更新全局变量 `faces`
    """
    global faces
    faces = parse_faces_data(data)  # 解析数据

def odom_callback(data):
    """
    Odom 话题回调函数，解析目标的位置和姿态
    """
    global target_position, target_orientation
    target_position = data.pose.pose.position
    target_orientation = data.pose.pose.orientation

def parse_faces_data(data):
    """
    解析 FovFaces 消息数据，返回面的顶点字典
    """
    parsed_faces = {}
    parsed_faces["face1"] = [{"x": p.x, "y": p.y, "z": p.z} for p in data.face1]
    parsed_faces["face2"] = [{"x": p.x, "y": p.y, "z": p.z} for p in data.face2]
    parsed_faces["face3"] = [{"x": p.x, "y": p.y, "z": p.z} for p in data.face3]
    parsed_faces["face4"] = [{"x": p.x, "y": p.y, "z": p.z} for p in data.face4]
    parsed_faces["face5"] = [{"x": p.x, "y": p.y, "z": p.z} for p in data.face5]
    return parsed_faces

def get_plane_equation(vertices):
    """
    计算平面方程 Ax + By + Cz + D = 0 的系数
    返回 (A, B, C, D)
    """
    if len(vertices) < 3:
        return None
    
    # 获取平面上的三个点
    p1 = np.array([vertices[0]["x"], vertices[0]["y"], vertices[0]["z"]])
    p2 = np.array([vertices[1]["x"], vertices[1]["y"], vertices[1]["z"]])
    p3 = np.array([vertices[2]["x"], vertices[2]["y"], vertices[2]["z"]])
    
    # 计算两个向量
    v1 = p2 - p1
    v2 = p3 - p1
    
    # 计算法向量 (A, B, C)
    normal = np.cross(v1, v2)
    A, B, C = normal
    
    # 计算 D
    D = -np.dot(normal, p1)
    
    return A, B, C, D

def point_to_plane_distance(point, plane_eq):
    """
    计算点到平面的距离
    """
    A, B, C, D = plane_eq
    x, y, z = point
    
    # 计算点到平面的有符号距离
    numerator = abs(A*x + B*y + C*z + D)
    denominator = np.sqrt(A**2 + B**2 + C**2)
    
    return numerator / denominator

def project_point_to_plane(point, plane_eq):
    """
    将点投影到平面上，返回投影点坐标
    """
    A, B, C, D = plane_eq
    x0, y0, z0 = point
    
    # 计算点到平面的有符号距离
    distance = (A*x0 + B*y0 + C*z0 + D) / (A**2 + B**2 + C**2)
    
    # 沿法向量方向投影点到平面
    x = x0 - distance * A
    y = y0 - distance * B
    z = z0 - distance * C
    
    return (x, y, z)

def is_point_in_polygon(point, vertices):
    """
    检查投影点是否在多边形内部
    使用射线法
    """
    # 将3D点和多边形转换为2D进行检查
    # 首先计算多边形所在平面
    plane_eq = get_plane_equation(vertices)
    
    # 判断点是否在多边形内的算法比较复杂
    # 这里使用简化方法：计算点到多边形中心的距离
    # 如果距离小于多边形的大致半径，则认为在内部
    
    # 计算多边形中心
    center_x = sum(v["x"] for v in vertices) / len(vertices)
    center_y = sum(v["y"] for v in vertices) / len(vertices)
    center_z = sum(v["z"] for v in vertices) / len(vertices)
    
    # 计算多边形半径（最远顶点到中心的距离）
    radius = max(np.sqrt((v["x"] - center_x)**2 + 
                        (v["y"] - center_y)**2 + 
                        (v["z"] - center_z)**2) for v in vertices)
    
    # 计算点到中心的距离
    px, py, pz = point
    dist = np.sqrt((px - center_x)**2 + (py - center_y)**2 + (pz - center_z)**2)
    
    # 如果距离小于半径，则认为在多边形内部
    return dist <= radius * 1.2  # 增加 20% 的容忍度

def find_nearest_intersection(target_pos, faces):
    """
    计算目标到每个FOV面的最近交点
    """
    if target_pos is None or not faces:
        return {}
    
    intersections = {}
    point = (target_pos.x, target_pos.y, target_pos.z)
    
    for face_name, vertices in faces.items():
        plane_eq = get_plane_equation(vertices)
        if plane_eq is None:
            continue
        
        # 计算目标点在平面上的投影
        proj_point = project_point_to_plane(point, plane_eq)
        
        # 检查投影点是否在多边形内
        if is_point_in_polygon(proj_point, vertices):
            # 计算距离
            distance = point_to_plane_distance(point, plane_eq)
            intersections[face_name] = {
                "point": proj_point,
                "distance": distance,
                "plane_eq": plane_eq  # 存储平面方程，用于后续高斯分布计算
            }
    
    return intersections

def generate_gaussian_distribution(center, plane_eq, radius=1.0):
    """
    在平面上生成高斯分布点
    
    参数:
    center - 高斯分布的中心点 (x, y, z)
    plane_eq - 平面方程系数 (A, B, C, D)
    radius - 生成点的范围半径
    
    返回:
    包含高斯分布点坐标的数组
    """
    # 计算平面的两个基向量
    A, B, C, D = plane_eq
    normal = np.array([A, B, C])
    normal = normal / np.linalg.norm(normal)  # 归一化法向量
    
    # 构造平面上的两个正交基向量
    if abs(normal[0]) < abs(normal[1]) and abs(normal[0]) < abs(normal[2]):
        v1 = np.array([0, normal[2], -normal[1]])
    elif abs(normal[1]) < abs(normal[2]):
        v1 = np.array([normal[2], 0, -normal[0]])
    else:
        v1 = np.array([normal[1], -normal[0], 0])
    
    v1 = v1 / np.linalg.norm(v1)  # 归一化
    v2 = np.cross(normal, v1)     # 第二个正交向量
    
    # 在平面上创建网格
    u = np.linspace(-radius, radius, gaussian_points)
    v = np.linspace(-radius, radius, gaussian_points)
    U, V = np.meshgrid(u, v)
    
    # 创建高斯分布的权重
    mean = np.array([0, 0])
    cov = np.array([[gaussian_std**2, 0], [0, gaussian_std**2]])
    rv = multivariate_normal(mean, cov)
    
    # 计算高斯分布的值
    pos = np.dstack((U, V))
    Z = rv.pdf(pos)
    Z = Z / Z.max()  # 归一化到 [0, 1]
    
    # 转换回3D坐标
    cx, cy, cz = center
    points = []
    
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            # 计算平面上的点坐标
            p = np.array([cx, cy, cz]) + U[i, j] * v1 + V[i, j] * v2
            points.append((p[0], p[1], p[2], Z[i, j]))
    
    return points

def update_plot(frame):
    """
    更新图形的函数，由 FuncAnimation 调用
    """
    global polys, ax, target_marker, target_position, target_orientation
    global intersection_markers, gaussian_plots

    # 清除之前的图形
    for poly in polys:
        poly.remove()
    polys.clear()
    
    # 清除之前的交点标记和高斯分布
    for marker in intersection_markers:
        marker.remove()
    intersection_markers.clear()
    
    for plot in gaussian_plots:
        plot.remove()
    gaussian_plots.clear()

    # 仅绘制每个面的轮廓
    for face_name, vertices in faces.items():
        x = [v["x"] for v in vertices]
        y = [v["y"] for v in vertices]
        z = [v["z"] for v in vertices]
        
        # 添加最后一个点连接回第一个点形成闭环
        x.append(x[0])
        y.append(y[0])
        z.append(z[0])
        
        # 绘制线条而不是面
        poly = ax.plot(x, y, z, color=line_color, linewidth=1.5)
        polys.extend(poly)

    # 更新目标位姿
    if target_position is not None:
        if target_marker is not None:
            target_marker.remove()  # 清除旧的标记
        # 绘制目标位置（点）
        target_marker = ax.scatter(target_position.x, target_position.y, target_position.z, color='black', s=100, label='Target')
        
        # 计算并显示交点
        intersections = find_nearest_intersection(target_position, faces)
        for face_name, data in intersections.items():
            point = data["point"]
            plane_eq = data["plane_eq"]
            x, y, z = point
            marker = ax.scatter(x, y, z, color=intersection_color, s=50)
            intersection_markers.append(marker)
            
            # 绘制从目标到交点的线
            line = ax.plot([target_position.x, x], [target_position.y, y], [target_position.z, z], 
                          'r--', linewidth=1.0, alpha=0.7)
            polys.extend(line)
            
            # 显示距离信息
            dist_text = ax.text(x, y, z, f"{data['distance']:.2f}m", color='red', fontsize=8)
            intersection_markers.append(dist_text)
            
            # 生成并绘制高斯分布
            radius = data["distance"] * 0.8  # 使用距离作为半径的参考
            gaussian_points_list = generate_gaussian_distribution(point, plane_eq, radius)
            
            # 提取点坐标和概率值
            xs = [p[0] for p in gaussian_points_list]
            ys = [p[1] for p in gaussian_points_list]
            zs = [p[2] for p in gaussian_points_list]
            probs = [p[3] for p in gaussian_points_list]
            
            # 使用颜色映射绘制高斯分布
            scatter = ax.scatter(xs, ys, zs, c=probs, cmap='Greens', alpha=0.5, s=25)
            gaussian_plots.append(scatter)
        
        # 如果需要显示姿态，可以绘制箭头
        if target_orientation is not None:
            # 将四元数转换为欧拉角
            roll, pitch, yaw = euler_from_quaternion([target_orientation.x, target_orientation.y, target_orientation.z, target_orientation.w])
            # 绘制箭头表示姿态
            arrow = ax.quiver(target_position.x, target_position.y, target_position.z,
                      np.cos(yaw), np.sin(yaw), 0, length=1.0, color='blue', normalize=True)
            polys.append(arrow)

    # 设置图形范围
    if faces:
        ax.set_xlim([min([v["x"] for face in faces.values() for v in face]) - 1,
                    max([v["x"] for face in faces.values() for v in face]) + 1])
        ax.set_ylim([min([v["y"] for face in faces.values() for v in face]) - 1,
                    max([v["y"] for face in faces.values() for v in face]) + 1])
        ax.set_zlim([min([v["z"] for face in faces.values() for v in face]) - 1,
                    max([v["z"] for face in faces.values() for v in face]) + 1])

    # 刷新图形
    plt.draw()

def main():
    """
    主函数，初始化 ROS 和 matplotlib
    """
    global fig, ax

    # 初始化 ROS 节点
    rospy.init_node('fov_visualization', anonymous=True)
    rospy.Subscriber('/drone0/odom_visualization/fov_faces', FovFaces, fov_callback)
    rospy.Subscriber('/target/odom', Odometry, odom_callback)

    # 初始化 matplotlib 图形
    plt.rcParams['axes.facecolor'] = '#f0f0f0'  # 设置背景颜色为浅灰色，便于观察
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('FOV and Target Visualization with Gaussian Distribution')

    # 添加图例
    ax.scatter([], [], [], color='black', s=100, label='Target')
    ax.scatter([], [], [], color=intersection_color, s=50, label='Intersection')
    ax.scatter([], [], [], c='green', alpha=0.5, s=25, label='Gaussian Distribution')
    ax.legend()

    # 使用 FuncAnimation 动态更新图形
    ani = FuncAnimation(fig, update_plot, interval=100)

    # 显示图形
    plt.show()
    # 保持 ROS 节点运行
    rospy.spin()

if __name__ == '__main__':
    main()