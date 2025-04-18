#ifndef _GRID_MAP_H
#define _GRID_MAP_H

#include <Eigen/Eigen>
#include <Eigen/StdVector>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <random>
#include <nav_msgs/Odometry.h>
#include <queue>
#include <ros/ros.h>
#include <tuple>
#include <visualization_msgs/Marker.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/time_synchronizer.h>

#include <quadrotor_msgs/FovFaces.h>
#include <plan_env/raycast.h>

#define logit(x) (log((x) / (1 - (x))))

using namespace std;

// voxel hashing
template <typename T>
struct matrix_hash : std::unary_function<T, size_t> {
  std::size_t operator()(T const& matrix) const {
    size_t seed = 0;
    for (size_t i = 0; i < matrix.size(); ++i) {
      auto elem = *(matrix.data() + i);
      seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }
    return seed;
  }
};

// constant parameters

struct MappingParameters {

  /* map properties */
  Eigen::Vector3d map_origin_, map_size_;
  Eigen::Vector3d map_min_boundary_, map_max_boundary_;  // map range in pos
  Eigen::Vector3i map_voxel_num_;                        // map range in index
  Eigen::Vector3d local_update_range_;
  double resolution_, resolution_inv_;
  double obstacles_inflation_;
  string frame_id_;
  int pose_type_;

  /* camera parameters */
  double cx_, cy_, fx_, fy_;

  /* time out */
  double odom_depth_timeout_;

  /* depth image projection filtering */
  double depth_filter_maxdist_, depth_filter_mindist_, depth_filter_tolerance_;
  int depth_filter_margin_;
  bool use_depth_filter_;
  double k_depth_scaling_factor_;
  int skip_pixel_;

  /* raycasting */
  double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  // occupancy probability
  double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_,
      min_occupancy_log_;                   // logit of occupancy probability
  double min_ray_length_, max_ray_length_;  // range of doing raycasting

  /* local map update and clear */
  int local_map_margin_;

  /* visualization and computation time display */
  double visualization_truncate_height_, virtual_ceil_height_, ground_height_, virtual_ceil_yp_, virtual_ceil_yn_;
  bool show_occ_time_;

  /* active mapping */
  double unknown_flag_;

  int esdf_x_bound_, esdf_y_bound_, esdf_z_bound_;
};

// intermediate mapping data for fusion

struct MappingData {
  // main map data, occupancy of each voxel and Euclidean distance

  std::vector<double> occupancy_buffer_;
  std::vector<char> occupancy_buffer_inflate_;

  std::vector<double> tmp_buffer1_;
  std::vector<double> tmp_buffer2_;
  std::vector<double> distance_buffer_;
  std::vector<double> distance_buffer_all_;
  std::vector<double> distance_buffer_neg_;
  std::vector<char> occupancy_buffer_neg;

  std::vector<double> freespace_tmp_buffer1_;
  std::vector<double> freespace_tmp_buffer2_;
  std::vector<double> freespace_distance_buffer_;
  std::vector<double> freespace_distance_buffer_all_;
  std::vector<double> freespace_distance_buffer_neg_;
  std::vector<char> freespace_occupancy_buffer_neg;

  Eigen::Vector3i min_esdf_;
  Eigen::Vector3i max_esdf_;
  Eigen::Vector3i freespace_min_esdf_;
  Eigen::Vector3i freespace_max_esdf_;

  int buffer_size_;

  // camera position and pose data

  Eigen::Vector3d camera_pos_, last_camera_pos_;
  Eigen::Matrix3d camera_r_m_, last_camera_r_m_;
  Eigen::Matrix4d cam2body_;
  // depth image data

  cv::Mat depth_image_, last_depth_image_;
  int image_cnt_;

  // flags of map state

  bool occ_need_update_, local_updated_, esdf_need_update_, freespace_need_update_;
  bool has_first_depth_;
  bool has_odom_, has_cloud_;

  // odom_depth_timeout_
  ros::Time last_occ_update_time_;
  bool flag_depth_odom_timeout_;
  bool flag_use_depth_fusion;

  // depth image projected point cloud

  vector<Eigen::Vector3d> proj_points_;
  int proj_points_cnt;

  // flag buffers for speeding up raycasting

  vector<short> count_hit_, count_hit_and_miss_;
  vector<char> flag_traverse_, flag_rayend_;
  char raycast_num_;
  queue<Eigen::Vector3i> cache_voxel_;

  // range of updating grid

  Eigen::Vector3i local_bound_min_, local_bound_max_;

  // computation time

  double fuse_time_, max_fuse_time_;
  int update_num_;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class GridMap {
public:
  GridMap() {}
  ~GridMap() {}

  enum { POSE_STAMPED = 1, ODOMETRY = 2, INVALID_IDX = -10000 };

  // occupancy map management
  void resetBuffer();
  void resetBuffer(Eigen::Vector3d min, Eigen::Vector3d max);

  inline void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
  inline void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
  inline int toAddress(const Eigen::Vector3i& id);
  inline int toAddress(int& x, int& y, int& z);
  inline bool isInMap(const Eigen::Vector3d& pos);
  inline bool isInMap(const Eigen::Vector3i& idx);

  inline void setOccupancy(Eigen::Vector3d pos, double occ = 1);
  inline void setOccupied(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3d pos);
  inline int getOccupancy(Eigen::Vector3i id);
  inline int getInflateOccupancy(Eigen::Vector3d pos);

  inline void boundIndex(Eigen::Vector3i& id);
  inline bool isESDFBound(int& x, int& y, int& z);
  inline bool isESDFBound_for_vis(int& x, int& y, int& z);
  inline bool isUnknown(const Eigen::Vector3i& id);
  inline bool isUnknown(const Eigen::Vector3d& pos);
  inline bool isKnownFree(const Eigen::Vector3i& id);
  inline bool isKnownOccupied(const Eigen::Vector3i& id);
  inline bool isKnownOccupied(const Eigen::Vector3d& pos);

  inline bool isInESDF(const Eigen::Vector3d& pos);
  inline bool isInESDF(const Eigen::Vector3i& idx);
  inline void esdfBoundIndex(Eigen::Vector3i& id);
  inline double getDistance(const Eigen::Vector3d& pos);

  bool evaluateESDF(const Eigen::Vector3d& pos, double& dist);

  bool evaluateESDFWithGrad(const Eigen::Vector3d& pos,
                                     double& dist, Eigen::Vector3d& grad);

  void getSurroundPts(const Eigen::Vector3d& pos, Eigen::Vector3d pts[2][2][2], Eigen::Vector3d& diff);

  void getSurroundDistance(Eigen::Vector3d pts[2][2][2], double dists[2][2][2]);

  void interpolateTrilinear(double values[2][2][2],
                            const Eigen::Vector3d& diff,
                            double& value);

  void interpolateTrilinear(double values[2][2][2],
                            const Eigen::Vector3d& diff,
                            double& value,
                            Eigen::Vector3d& grad);


  bool evaluateVisibilitySDFWithGrad(const Eigen::Vector3d& pos, const Eigen::Vector3d& target,
                                            double& dist, Eigen::Vector3d& grad);
                                      
  void getSurroundVisibility(Eigen::Vector3d pts[2][2][2], double visibilities[2][2][2], const Eigen::Vector3d& target);
  double getVisibility(Eigen::Vector3d pos, const Eigen::Vector3d& target);
  double getVisibilityWithESDFGradient(Eigen::Vector3d pos, const Eigen::Vector3d& target, Eigen::Vector3d &grad);
  bool evaluateVisibilitySDFInTheEnd(Eigen::Vector3d pos, Eigen::Vector3d target);


  void initMap(ros::NodeHandle& nh);

  void publishMap();
  void publishMapInflate(bool all_info = false);
  void publishFreespace();
  void publishESDF();
  void publishVisibilitySDF();

  void publishDepth();

  bool hasDepthObservation();
  bool odomValid();
  void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
  inline double getResolution();
  inline double getResolution_inv();

  Eigen::Vector3d getOrigin();
  int getVoxelNum();
  bool getOdomDepthTimeout() { return md_.flag_depth_odom_timeout_; }

  typedef std::shared_ptr<GridMap> Ptr;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
  MappingParameters mp_;
  MappingData md_;

  // get depth image and camera pose
  void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
                         const geometry_msgs::PoseStampedConstPtr& pose);
  void extrinsicCallback(const nav_msgs::OdometryConstPtr& odom);
  void depthOdomCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& odom);
  void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& img);
  void odomCallback(const nav_msgs::OdometryConstPtr& odom);
  
  void fovFacesCallback(const quadrotor_msgs::FovFaces::ConstPtr& msg);

  // update occupancy by raycasting
  void updateOccupancyCallback(const ros::TimerEvent& /*event*/);
  void visCallback(const ros::TimerEvent& /*event*/);
  void updateESDFCallback(const ros::TimerEvent& /*event*/);
  void updateFreespaceCallback(const ros::TimerEvent& /*event*/);
  void updateESDF3d();
  void updateFreespaceESDF3d();

  template <typename F_get_val, typename F_set_val>
  void fillESDF(F_get_val f_get_val, F_set_val f_set_val, int start, int end, int dim);

  // main update process
  void projectDepthImage();
  void raycastProcess();
  void clearAndInflateLocalMap();

  inline void inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts);
  int setCacheOccupancy(Eigen::Vector3d pos, int occ);
  Eigen::Vector3d closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt);

  // typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
  // nav_msgs::Odometry> SyncPolicyImageOdom; typedef
  // message_filters::sync_policies::ExactTime<sensor_msgs::Image,
  // geometry_msgs::PoseStamped> SyncPolicyImagePose;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
      SyncPolicyImageOdom;
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
      SyncPolicyImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
  typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;

  ros::NodeHandle node_;
  shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
  shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
  shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
  SynchronizerImagePose sync_image_pose_;
  SynchronizerImageOdom sync_image_odom_;

  ros::Subscriber indep_cloud_sub_, indep_odom_sub_, extrinsic_sub_;
  ros::Publisher map_pub_, map_inf_pub_, map_freespace_pub_, map_esdf_pub_, visibility_esdf_pub_;
  ros::Timer occ_timer_, ESDF_timer_, vis_timer_, freespace_timer_;
  
  ros::Publisher esdf_pub_;
  ros::Subscriber fov_faves_sub_;

  uniform_real_distribution<double> rand_noise_;
  normal_distribution<double> rand_noise2_;
  default_random_engine eng_;
};

/* ============================== definition of inline function
 * ============================== */

inline int GridMap::toAddress(const Eigen::Vector3i& id) {
  return id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + id(1) * mp_.map_voxel_num_(2) + id(2);
}

inline int GridMap::toAddress(int& x, int& y, int& z) {
  return x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + y * mp_.map_voxel_num_(2) + z;
}

inline void GridMap::boundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), mp_.map_voxel_num_(0) - 1), 0);
  id1(1) = max(min(id(1), mp_.map_voxel_num_(1) - 1), 0);
  id1(2) = max(min(id(2), mp_.map_voxel_num_(2) - 1), 0);
  id = id1;
}

inline bool GridMap::isESDFBound(int& x, int& y, int& z) {
  if( x == md_.min_esdf_(0) || x == md_.max_esdf_(0) )
    return true;

  if( y == md_.min_esdf_(1) || y == md_.max_esdf_(1) )
    return true;

  if( z == md_.min_esdf_(2) || z == md_.max_esdf_(2) )
    return true;
  
  return false;
}

inline bool GridMap::isESDFBound_for_vis(int& x, int& y, int& z) {
  if( x == md_.min_esdf_(0) || x == md_.max_esdf_(0) )
    return true;

  if( y == md_.min_esdf_(1) || y == md_.max_esdf_(1) )
    return true;
  
  return false;
}

inline bool GridMap::isUnknown(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  return md_.occupancy_buffer_[toAddress(id1)] < mp_.clamp_min_log_ - 1e-3;
}

inline bool GridMap::isUnknown(const Eigen::Vector3d& pos) {
  Eigen::Vector3i idc;
  posToIndex(pos, idc);
  return isUnknown(idc);
}



inline bool GridMap::isKnownFree(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);

  // return md_.occupancy_buffer_[adr] >= mp_.clamp_min_log_ &&
  //     md_.occupancy_buffer_[adr] < mp_.min_occupancy_log_;
  return md_.occupancy_buffer_[adr] >= mp_.clamp_min_log_ && md_.occupancy_buffer_inflate_[adr] == 0;
}

inline bool GridMap::isKnownOccupied(const Eigen::Vector3i& id) {
  Eigen::Vector3i id1 = id;
  boundIndex(id1);
  int adr = toAddress(id1);

  return md_.occupancy_buffer_inflate_[adr] == 1;
}

inline bool GridMap::isKnownOccupied(const Eigen::Vector3d& pos) {
  Eigen::Vector3d pos1 = pos;
  Eigen::Vector3i idc;
  posToIndex(pos, idc);
  boundIndex(idc);
  int adr = toAddress(idc);

  return md_.occupancy_buffer_inflate_[adr] == 1;
}

inline void GridMap::setOccupied(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  md_.occupancy_buffer_inflate_[id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
                                id(1) * mp_.map_voxel_num_(2) + id(2)] = 1;
}

inline void GridMap::setOccupancy(Eigen::Vector3d pos, double occ) {
  if (occ != 1 && occ != 0) {
    cout << "occ value error!" << endl;
    return;
  }

  if (!isInMap(pos)) return;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  md_.occupancy_buffer_[toAddress(id)] = occ;
}


inline bool GridMap::isInESDF(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);

  return isInESDF(id);
}

inline bool GridMap::isInESDF(const Eigen::Vector3i& idx) {
  if (idx(0) < (md_.min_esdf_(0)+2) || idx(1) < (md_.min_esdf_(1)+2) || idx(2) < (md_.min_esdf_(2)+1)) {
    return false;
  }
  if (idx(0) > (md_.max_esdf_(0) - 2) || idx(1) > (md_.max_esdf_(1) - 2) ||
      idx(2) > (md_.max_esdf_(2) - 1)) {
    return false;
  }
  return true;
}

inline void GridMap::esdfBoundIndex(Eigen::Vector3i& id) {
  Eigen::Vector3i id1;
  id1(0) = max(min(id(0), md_.max_esdf_(0) - 1), md_.min_esdf_(0));
  id1(1) = max(min(id(1), md_.max_esdf_(1) - 1), md_.min_esdf_(1));
  id1(2) = max(min(id(2), md_.max_esdf_(2) - 1), md_.min_esdf_(2));
  id = id1;
}

// free space esdf
inline double GridMap::getDistance(const Eigen::Vector3d& pos) {
  Eigen::Vector3i id;
  posToIndex(pos, id);
  esdfBoundIndex(id);

  return md_.distance_buffer_all_[toAddress(id)];
}

inline int GridMap::getOccupancy(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return -1;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  return md_.occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

inline int GridMap::getInflateOccupancy(Eigen::Vector3d pos) {
  if (!isInMap(pos)) return -1;

  Eigen::Vector3i id;
  posToIndex(pos, id);

  return int(md_.occupancy_buffer_inflate_[toAddress(id)]);
}

inline int GridMap::getOccupancy(Eigen::Vector3i id) {
  if (id(0) < 0 || id(0) >= mp_.map_voxel_num_(0) || id(1) < 0 || id(1) >= mp_.map_voxel_num_(1) ||
      id(2) < 0 || id(2) >= mp_.map_voxel_num_(2))
    return -1;

  return md_.occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
}

inline bool GridMap::isInMap(const Eigen::Vector3d& pos) {
  if (pos(0) < mp_.map_min_boundary_(0) + 1e-4 || pos(1) < mp_.map_min_boundary_(1) + 1e-4 ||
      pos(2) < mp_.map_min_boundary_(2) + 1e-4) {
    // cout << "less than min range!" << endl;
    return false;
  }
  if (pos(0) > mp_.map_max_boundary_(0) - 1e-4 || pos(1) > mp_.map_max_boundary_(1) - 1e-4 ||
      pos(2) > mp_.map_max_boundary_(2) - 1e-4) {
    return false;
  }
  return true;
}

inline bool GridMap::isInMap(const Eigen::Vector3i& idx) {
  if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) {
    return false;
  }
  if (idx(0) > mp_.map_voxel_num_(0) - 1 || idx(1) > mp_.map_voxel_num_(1) - 1 ||
      idx(2) > mp_.map_voxel_num_(2) - 1) {
    return false;
  }
  return true;
}

inline void GridMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
  for (int i = 0; i < 3; ++i) id(i) = floor((pos(i) - mp_.map_origin_(i)) * mp_.resolution_inv_);
}

inline void GridMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
  for (int i = 0; i < 3; ++i) pos(i) = (id(i) + 0.5) * mp_.resolution_ + mp_.map_origin_(i);
}

inline void GridMap::inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts) {
  int num = 0;

  /* ---------- + shape inflate ---------- */
  // for (int x = -step; x <= step; ++x)
  // {
  //   if (x == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1), pt(2));
  // }
  // for (int y = -step; y <= step; ++y)
  // {
  //   if (y == 0)
  //     continue;
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1) + y, pt(2));
  // }
  // for (int z = -1; z <= 1; ++z)
  // {
  //   pts[num++] = Eigen::Vector3i(pt(0), pt(1), pt(2) + z);
  // }

  /* ---------- all inflate ---------- */
  for (int x = -step; x <= step; ++x)
    for (int y = -step; y <= step; ++y)
      for (int z = -step; z <= step; ++z) {
        pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
      }
}

inline double GridMap::getResolution() { return mp_.resolution_; }
inline double GridMap::getResolution_inv() { return mp_.resolution_inv_; }

#endif

// #ifndef _GRID_MAP_H
// #define _GRID_MAP_H

// #include <Eigen/Eigen>
// #include <Eigen/StdVector>
// #include <cv_bridge/cv_bridge.h>
// #include <geometry_msgs/PoseStamped.h>
// #include <iostream>
// #include <random>
// #include <nav_msgs/Odometry.h>
// #include <queue>
// #include <ros/ros.h>
// #include <tuple>
// #include <visualization_msgs/Marker.h>

// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>
// #include <pcl_conversions/pcl_conversions.h>

// #include <message_filters/subscriber.h>
// #include <message_filters/sync_policies/approximate_time.h>
// #include <message_filters/sync_policies/exact_time.h>
// #include <message_filters/time_synchronizer.h>

// #include <plan_env/raycast.h>

// #define logit(x) (log((x) / (1 - (x))))

// using namespace std;

// // voxel hashing
// template <typename T>
// struct matrix_hash : std::unary_function<T, size_t> {
//   std::size_t operator()(T const& matrix) const {
//     size_t seed = 0;
//     for (size_t i = 0; i < matrix.size(); ++i) {
//       auto elem = *(matrix.data() + i);
//       seed ^= std::hash<typename T::Scalar>()(elem) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
//     }
//     return seed;
//   }
// };

// // constant parameters

// struct MappingParameters {

//   /* map properties */
//   Eigen::Vector3d map_origin_, map_size_;
//   Eigen::Vector3d map_min_boundary_, map_max_boundary_;  // map range in pos
//   Eigen::Vector3i map_voxel_num_;                        // map range in index
//   Eigen::Vector3d local_update_range_;
//   double resolution_, resolution_inv_;
//   double obstacles_inflation_;
//   string frame_id_;
//   int pose_type_;

//   /* camera parameters */
//   double cx_, cy_, fx_, fy_;

//   /* depth image projection filtering */
//   double depth_filter_maxdist_, depth_filter_mindist_, depth_filter_tolerance_;
//   int depth_filter_margin_;
//   bool use_depth_filter_;
//   double k_depth_scaling_factor_;
//   int skip_pixel_;

//   /* raycasting */
//   double p_hit_, p_miss_, p_min_, p_max_, p_occ_;  // occupancy probability
//   double prob_hit_log_, prob_miss_log_, clamp_min_log_, clamp_max_log_,
//       min_occupancy_log_;                   // logit of occupancy probability
//   double min_ray_length_, max_ray_length_;  // range of doing raycasting

//   /* local map update and clear */
//   int local_map_margin_;

//   /* visualization and computation time display */
//   double visualization_truncate_height_, virtual_ceil_height_, ground_height_;
//   bool show_occ_time_;

//   /* active mapping */
//   double unknown_flag_;
// };

// // intermediate mapping data for fusion

// struct MappingData {
//   // main map data, occupancy of each voxel and Euclidean distance

//   std::vector<double> occupancy_buffer_;
//   std::vector<char> occupancy_buffer_inflate_;

//   // camera position and pose data

//   Eigen::Vector3d camera_pos_, last_camera_pos_;
//   Eigen::Quaterniond camera_q_, last_camera_q_;

//   // depth image data

//   cv::Mat depth_image_, last_depth_image_;
//   int image_cnt_;

//   // flags of map state

//   bool occ_need_update_, local_updated_;
//   bool has_first_depth_;
//   bool has_odom_, has_cloud_;

//   // depth image projected point cloud

//   vector<Eigen::Vector3d> proj_points_;
//   int proj_points_cnt;

//   // flag buffers for speeding up raycasting

//   vector<short> count_hit_, count_hit_and_miss_;
//   vector<char> flag_traverse_, flag_rayend_;
//   char raycast_num_;
//   queue<Eigen::Vector3i> cache_voxel_;

//   // range of updating grid

//   Eigen::Vector3i local_bound_min_, local_bound_max_;

//   // computation time

//   double fuse_time_, max_fuse_time_;
//   int update_num_;

//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW
// };

// class GridMap {
// public:
//   GridMap() {}
//   ~GridMap() {}

//   enum { POSE_STAMPED = 1, ODOMETRY = 2, INVALID_IDX = -10000 };

//   // occupancy map management
//   void resetBuffer();
//   void resetBuffer(Eigen::Vector3d min, Eigen::Vector3d max);

//   inline void posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id);
//   inline void indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos);
//   inline int toAddress(const Eigen::Vector3i& id);
//   inline int toAddress(int& x, int& y, int& z);
//   inline bool isInMap(const Eigen::Vector3d& pos);
//   inline bool isInMap(const Eigen::Vector3i& idx);

//   inline void setOccupancy(Eigen::Vector3d pos, double occ = 1);
//   inline void setOccupied(Eigen::Vector3d pos);
//   inline int getOccupancy(Eigen::Vector3d pos);
//   inline int getOccupancy(Eigen::Vector3i id);
//   inline int getInflateOccupancy(Eigen::Vector3d pos);

//   inline void boundIndex(Eigen::Vector3i& id);
//   inline bool isUnknown(const Eigen::Vector3i& id);
//   inline bool isUnknown(const Eigen::Vector3d& pos);
//   inline bool isKnownFree(const Eigen::Vector3i& id);
//   inline bool isKnownOccupied(const Eigen::Vector3i& id);

//   void initMap(ros::NodeHandle& nh);

//   void publishMap();
//   void publishMapInflate(bool all_info = false);

//   void publishUnknown();
//   void publishDepth();

//   bool hasDepthObservation();
//   bool odomValid();
//   void getRegion(Eigen::Vector3d& ori, Eigen::Vector3d& size);
//   inline double getResolution();
//   Eigen::Vector3d getOrigin();
//   int getVoxelNum();

//   typedef std::shared_ptr<GridMap> Ptr;

//   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

// private:
//   MappingParameters mp_;
//   MappingData md_;

//   // get depth image and camera pose
//   void depthPoseCallback(const sensor_msgs::ImageConstPtr& img,
//                          const geometry_msgs::PoseStampedConstPtr& pose);
//   void depthOdomCallback(const sensor_msgs::ImageConstPtr& img, const nav_msgs::OdometryConstPtr& odom);
//   void cloudCallback(const sensor_msgs::PointCloud2ConstPtr& img);
//   void odomCallback(const nav_msgs::OdometryConstPtr& odom);

//   // update occupancy by raycasting
//   void updateOccupancyCallback(const ros::TimerEvent& /*event*/);
//   void visCallback(const ros::TimerEvent& /*event*/);

//   // main update process
//   void projectDepthImage();
//   void raycastProcess();
//   void clearAndInflateLocalMap();

//   inline void inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts);
//   int setCacheOccupancy(Eigen::Vector3d pos, int occ);
//   Eigen::Vector3d closetPointInMap(const Eigen::Vector3d& pt, const Eigen::Vector3d& camera_pt);

//   // typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image,
//   // nav_msgs::Odometry> SyncPolicyImageOdom; typedef
//   // message_filters::sync_policies::ExactTime<sensor_msgs::Image,
//   // geometry_msgs::PoseStamped> SyncPolicyImagePose;
//   typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, nav_msgs::Odometry>
//       SyncPolicyImageOdom;
//   typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, geometry_msgs::PoseStamped>
//       SyncPolicyImagePose;
//   typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImagePose>> SynchronizerImagePose;
//   typedef shared_ptr<message_filters::Synchronizer<SyncPolicyImageOdom>> SynchronizerImageOdom;

//   ros::NodeHandle node_;
//   shared_ptr<message_filters::Subscriber<sensor_msgs::Image>> depth_sub_;
//   shared_ptr<message_filters::Subscriber<geometry_msgs::PoseStamped>> pose_sub_;
//   shared_ptr<message_filters::Subscriber<nav_msgs::Odometry>> odom_sub_;
//   SynchronizerImagePose sync_image_pose_;
//   SynchronizerImageOdom sync_image_odom_;

//   ros::Subscriber indep_cloud_sub_, indep_odom_sub_;
//   ros::Publisher map_pub_, map_inf_pub_;
//   ros::Publisher unknown_pub_;
//   ros::Timer occ_timer_, vis_timer_;

//   //
//   uniform_real_distribution<double> rand_noise_;
//   normal_distribution<double> rand_noise2_;
//   default_random_engine eng_;
// };

// /* ============================== definition of inline function
//  * ============================== */

// inline int GridMap::toAddress(const Eigen::Vector3i& id) {
//   return id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + id(1) * mp_.map_voxel_num_(2) + id(2);
// }

// inline int GridMap::toAddress(int& x, int& y, int& z) {
//   return x * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) + y * mp_.map_voxel_num_(2) + z;
// }

// inline void GridMap::boundIndex(Eigen::Vector3i& id) {
//   Eigen::Vector3i id1;
//   id1(0) = max(min(id(0), mp_.map_voxel_num_(0) - 1), 0);
//   id1(1) = max(min(id(1), mp_.map_voxel_num_(1) - 1), 0);
//   id1(2) = max(min(id(2), mp_.map_voxel_num_(2) - 1), 0);
//   id = id1;
// }

// inline bool GridMap::isUnknown(const Eigen::Vector3i& id) {
//   Eigen::Vector3i id1 = id;
//   boundIndex(id1);
//   return md_.occupancy_buffer_[toAddress(id1)] < mp_.clamp_min_log_ - 1e-3;
// }

// inline bool GridMap::isUnknown(const Eigen::Vector3d& pos) {
//   Eigen::Vector3i idc;
//   posToIndex(pos, idc);
//   return isUnknown(idc);
// }

// inline bool GridMap::isKnownFree(const Eigen::Vector3i& id) {
//   Eigen::Vector3i id1 = id;
//   boundIndex(id1);
//   int adr = toAddress(id1);

//   // return md_.occupancy_buffer_[adr] >= mp_.clamp_min_log_ &&
//   //     md_.occupancy_buffer_[adr] < mp_.min_occupancy_log_;
//   return md_.occupancy_buffer_[adr] >= mp_.clamp_min_log_ && md_.occupancy_buffer_inflate_[adr] == 0;
// }

// inline bool GridMap::isKnownOccupied(const Eigen::Vector3i& id) {
//   Eigen::Vector3i id1 = id;
//   boundIndex(id1);
//   int adr = toAddress(id1);

//   return md_.occupancy_buffer_inflate_[adr] == 1;
// }

// inline void GridMap::setOccupied(Eigen::Vector3d pos) {
//   if (!isInMap(pos)) return;

//   Eigen::Vector3i id;
//   posToIndex(pos, id);

//   md_.occupancy_buffer_inflate_[id(0) * mp_.map_voxel_num_(1) * mp_.map_voxel_num_(2) +
//                                 id(1) * mp_.map_voxel_num_(2) + id(2)] = 1;
// }

// inline void GridMap::setOccupancy(Eigen::Vector3d pos, double occ) {
//   if (occ != 1 && occ != 0) {
//     cout << "occ value error!" << endl;
//     return;
//   }

//   if (!isInMap(pos)) return;

//   Eigen::Vector3i id;
//   posToIndex(pos, id);

//   md_.occupancy_buffer_[toAddress(id)] = occ;
// }

// inline int GridMap::getOccupancy(Eigen::Vector3d pos) {
//   if (!isInMap(pos)) return -1;

//   Eigen::Vector3i id;
//   posToIndex(pos, id);

//   return md_.occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
// }

// inline int GridMap::getInflateOccupancy(Eigen::Vector3d pos) {
//   if (!isInMap(pos)) return -1;

//   Eigen::Vector3i id;
//   posToIndex(pos, id);

//   return int(md_.occupancy_buffer_inflate_[toAddress(id)]);
// }

// inline int GridMap::getOccupancy(Eigen::Vector3i id) {
//   if (id(0) < 0 || id(0) >= mp_.map_voxel_num_(0) || id(1) < 0 || id(1) >= mp_.map_voxel_num_(1) ||
//       id(2) < 0 || id(2) >= mp_.map_voxel_num_(2))
//     return -1;

//   return md_.occupancy_buffer_[toAddress(id)] > mp_.min_occupancy_log_ ? 1 : 0;
// }

// inline bool GridMap::isInMap(const Eigen::Vector3d& pos) {
//   if (pos(0) < mp_.map_min_boundary_(0) + 1e-4 || pos(1) < mp_.map_min_boundary_(1) + 1e-4 ||
//       pos(2) < mp_.map_min_boundary_(2) + 1e-4) {
//     // cout << "less than min range!" << endl;
//     return false;
//   }
//   if (pos(0) > mp_.map_max_boundary_(0) - 1e-4 || pos(1) > mp_.map_max_boundary_(1) - 1e-4 ||
//       pos(2) > mp_.map_max_boundary_(2) - 1e-4) {
//     return false;
//   }
//   return true;
// }

// inline bool GridMap::isInMap(const Eigen::Vector3i& idx) {
//   if (idx(0) < 0 || idx(1) < 0 || idx(2) < 0) {
//     return false;
//   }
//   if (idx(0) > mp_.map_voxel_num_(0) - 1 || idx(1) > mp_.map_voxel_num_(1) - 1 ||
//       idx(2) > mp_.map_voxel_num_(2) - 1) {
//     return false;
//   }
//   return true;
// }

// inline void GridMap::posToIndex(const Eigen::Vector3d& pos, Eigen::Vector3i& id) {
//   for (int i = 0; i < 3; ++i) id(i) = floor((pos(i) - mp_.map_origin_(i)) * mp_.resolution_inv_);
// }

// inline void GridMap::indexToPos(const Eigen::Vector3i& id, Eigen::Vector3d& pos) {
//   for (int i = 0; i < 3; ++i) pos(i) = (id(i) + 0.5) * mp_.resolution_ + mp_.map_origin_(i);
// }

// inline void GridMap::inflatePoint(const Eigen::Vector3i& pt, int step, vector<Eigen::Vector3i>& pts) {
//   int num = 0;

//   /* ---------- + shape inflate ---------- */
//   // for (int x = -step; x <= step; ++x)
//   // {
//   //   if (x == 0)
//   //     continue;
//   //   pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1), pt(2));
//   // }
//   // for (int y = -step; y <= step; ++y)
//   // {
//   //   if (y == 0)
//   //     continue;
//   //   pts[num++] = Eigen::Vector3i(pt(0), pt(1) + y, pt(2));
//   // }
//   // for (int z = -1; z <= 1; ++z)
//   // {
//   //   pts[num++] = Eigen::Vector3i(pt(0), pt(1), pt(2) + z);
//   // }

//   /* ---------- all inflate ---------- */
//   for (int x = -step; x <= step; ++x)
//     for (int y = -step; y <= step; ++y)
//       for (int z = -step; z <= step; ++z) {
//         pts[num++] = Eigen::Vector3i(pt(0) + x, pt(1) + y, pt(2) + z);
//       }
// }

// inline double GridMap::getResolution() { return mp_.resolution_; }

// #endif