
#include <plan_manage/ego_replan_fsm.h>

namespace ego_planner
{

  void EGOReplanFSM::init(ros::NodeHandle &nh)
  {
    current_wp_ = 0;
    exec_state_ = FSM_EXEC_STATE::INIT;
    have_target_ = false;
    have_odom_ = false;
    have_recv_pre_agent_ = false;
    receive_target_traj_ = false;

    /*  fsm param  */
    nh.param("fsm/flight_type", target_type_, -1);
    nh.param("fsm/thresh_replan_time", replan_thresh_, -1.0);
    nh.param("fsm/thresh_no_replan_meter", no_replan_thresh_, -1.0);
    nh.param("fsm/planning_horizon", planning_horizen_, -1.0);
    nh.param("fsm/planning_horizen_time", planning_horizen_time_, -1.0);
    nh.param("fsm/emergency_time", emergency_time_, 1.0);
    nh.param("fsm/realworld_experiment", flag_realworld_experiment_, false);
    nh.param("fsm/fail_safe", enable_fail_safe_, true);

    have_trigger_ = !flag_realworld_experiment_;

    nh.param("fsm/waypoint_num", waypoint_num_, -1);
    for (int i = 0; i < waypoint_num_; i++)
    {
      nh.param("fsm/waypoint" + to_string(i) + "_x", waypoints_[i][0], -1.0);
      nh.param("fsm/waypoint" + to_string(i) + "_y", waypoints_[i][1], -1.0);
      nh.param("fsm/waypoint" + to_string(i) + "_z", waypoints_[i][2], -1.0);
    }

    /* initialize main modules */
    visualization_.reset(new PlanningVisualization(nh));
    planner_manager_.reset(new EGOPlannerManager);
    planner_manager_->initPlanModules(nh, visualization_);
    planner_manager_->deliverTrajToOptimizer(); // store trajectories
    planner_manager_->setDroneIdtoOpt();

    /* callback */
    exec_timer_ = nh.createTimer(ros::Duration(0.01), &EGOReplanFSM::execFSMCallback, this);
    safety_timer_ = nh.createTimer(ros::Duration(0.05), &EGOReplanFSM::checkCollisionCallback, this);

    odom_sub_ = nh.subscribe("odom_world", 1, &EGOReplanFSM::odometryCallback, this, ros::TransportHints().tcpNoDelay());

    if (planner_manager_->pp_.drone_id >= 1)
    {
      string sub_topic_name = string("/drone_") + std::to_string(planner_manager_->pp_.drone_id - 1) + string("_planning/swarm_trajs");
      swarm_trajs_sub_ = nh.subscribe(sub_topic_name.c_str(), 10, &EGOReplanFSM::swarmTrajsCallback, this, ros::TransportHints().tcpNoDelay());
    }
    string pub_topic_name = string("/drone_") + std::to_string(planner_manager_->pp_.drone_id) + string("_planning/swarm_trajs");
    swarm_trajs_pub_ = nh.advertise<traj_utils::MultiBsplines>(pub_topic_name.c_str(), 10);

    // broadcast_bspline_pub_ = nh.advertise<traj_utils::Bspline>("planning/broadcast_bspline_from_planner", 10);
    broadcast_bspline_sub_ = nh.subscribe("planning/broadcast_bspline_to_planner", 100, &EGOReplanFSM::BroadcastBsplineCallback, this, ros::TransportHints().tcpNoDelay());

    bspline_pub_ = nh.advertise<traj_utils::Bspline>("planning/bspline", 10);
    data_disp_pub_ = nh.advertise<traj_utils::DataDisp>("planning/data_display", 100);

    // for visual traj
    pos_list_pub_ = nh.advertise<visualization_msgs::Marker>("traj_pos_list", 2);
    cpt_list_pub_ = nh.advertise<visualization_msgs::Marker>("control_points_list", 2);
    yaw_list_pub_ = nh.advertise<visualization_msgs::MarkerArray>("traj_yaw_list", 2);
    new_predict_list_pub_ = nh.advertise<visualization_msgs::Marker>("pred_control_points_list", 2);

    // debug for optimize
    debug_cpt_list_pub_ = nh.advertise<visualization_msgs::Marker>("debug_control_points_list", 2);
    attract_score_list_pub_ = nh.advertise<visualization_msgs::MarkerArray>("track_line", 2);
    gradient_list_pub_ = nh.advertise<visualization_msgs::MarkerArray>("gradient_line", 2);
    pk_grad_list_pub_ = nh.advertise<visualization_msgs::MarkerArray>("pk_gradient_line", 2);

    if (target_type_ == TARGET_TYPE::MANUAL_TARGET)
    {
      waypoint_sub_ = nh.subscribe("waypoint_generator/waypoints", 1, &EGOReplanFSM::waypointCallback, this);
    }
    else if (target_type_ == TARGET_TYPE::PRESET_TARGET)
    {
      trigger_sub_ = nh.subscribe("traj_start_trigger", 1, &EGOReplanFSM::triggerCallback, this);
      ros::Duration(1.0).sleep();
      ROS_WARN("Waiting for trigger from [n3ctrl] from RC");
      while (ros::ok() && (!have_odom_ || !have_trigger_))
      {
        ros::spinOnce();
        ros::Duration(0.001).sleep();
      }
      planGlobalTrajbyGivenWps();
    }
    else
      cout << "Wrong target_type_ value! target_type_=" << target_type_ << endl;
  }

  void EGOReplanFSM::planGlobalTrajbyGivenWps()
  {
    std::vector<Eigen::Vector3d> wps(waypoint_num_);
    for (int i = 0; i < waypoint_num_; i++)
    {
      wps[i](0) = waypoints_[i][0];
      wps[i](1) = waypoints_[i][1];
      wps[i](2) = waypoints_[i][2];

      end_pt_ = wps.back();
    }
    bool success = planner_manager_->planGlobalTrajWaypoints(odom_pos_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), wps, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    for (size_t i = 0; i < (size_t)waypoint_num_; i++)
    {
      visualization_->displayGoalPoint(wps[i], Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, i);
      ros::Duration(0.001).sleep();
    }

    if (success)
    {

      /*** display ***/
      constexpr double step_size_t = 0.1;
      int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
      std::vector<Eigen::Vector3d> gloabl_traj(i_end);
      for (int i = 0; i < i_end; i++)
      {
        gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
      }

      end_vel_.setZero();
      have_target_ = true;
      have_new_target_ = true;

      /*** FSM ***/
      if (exec_state_ == WAIT_TARGET)
        changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
      else if (exec_state_ == EXEC_TRAJ)
        changeFSMExecState(REPLAN_TRAJ, "TRIG");

      // trigger_ = true;

      // visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(1, 0, 0, 1), 0.3, 0);
      ros::Duration(0.001).sleep();
      visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
      ros::Duration(0.001).sleep();
    }
    else
    {
      ROS_ERROR("Unable to generate global trajectory!");
    }
  }

  void EGOReplanFSM::triggerCallback(const geometry_msgs::PoseStampedPtr &msg)
  {
    have_trigger_ = true;
    cout << "Triggered!" << endl;
    init_pt_ = odom_pos_;
  }

  void EGOReplanFSM::waypointCallback(const nav_msgs::PathConstPtr &msg)
  {
    if (msg->poses[0].pose.position.z < -0.1)
      return;

    // cout << "Triggered!" << endl;

    // the tracker should be triggered by the target
    return;
    // trigger_ = true;
    init_pt_ = odom_pos_;

    bool success = false;
    end_pt_ << msg->poses[0].pose.position.x, msg->poses[0].pose.position.y, 1.0;
    success = planner_manager_->planGlobalTraj(odom_pos_, odom_vel_, Eigen::Vector3d::Zero(), end_pt_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());

    visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(0, 0.5, 0.5, 1), 0.3, 0);

    if (success)
    {

      /*** display ***/
      constexpr double step_size_t = 0.1;
      int i_end = floor(planner_manager_->global_data_.global_duration_ / step_size_t);
      vector<Eigen::Vector3d> gloabl_traj(i_end);
      for (int i = 0; i < i_end; i++)
      {
        gloabl_traj[i] = planner_manager_->global_data_.global_traj_.evaluate(i * step_size_t);
      }

      end_vel_.setZero();
      have_target_ = true;
      have_new_target_ = true;

      /*** FSM ***/
      // if (exec_state_ == WAIT_TARGET)
      //   changeFSMExecState(GEN_NEW_TRAJ, "TRIG");
      // else if (exec_state_ == EXEC_TRAJ)
      //   changeFSMExecState(REPLAN_TRAJ, "TRIG");

      // visualization_->displayGoalPoint(end_pt_, Eigen::Vector4d(1, 0, 0, 1), 0.3, 0);
      visualization_->displayGlobalPathList(gloabl_traj, 0.1, 0);
    }
    else
    {
      ROS_ERROR("Unable to generate global trajectory!");
    }
    cout << "A3" << endl;
  }

  void EGOReplanFSM::odometryCallback(const nav_msgs::OdometryConstPtr &msg)
  {
    odom_pos_(0) = msg->pose.pose.position.x;
    odom_pos_(1) = msg->pose.pose.position.y;
    odom_pos_(2) = msg->pose.pose.position.z;

    odom_vel_(0) = msg->twist.twist.linear.x;
    odom_vel_(1) = msg->twist.twist.linear.y;
    odom_vel_(2) = msg->twist.twist.linear.z;

    //odom_acc_ = estimateAcc( msg );

    odom_orient_.w() = msg->pose.pose.orientation.w;
    odom_orient_.x() = msg->pose.pose.orientation.x;
    odom_orient_.y() = msg->pose.pose.orientation.y;
    odom_orient_.z() = msg->pose.pose.orientation.z;

    have_odom_ = true;
  }

  void EGOReplanFSM::BroadcastBsplineCallback(const traj_utils::BsplinePtr &msg)
  {
    size_t id = msg->drone_id;
    if ( (int)id == planner_manager_->pp_.drone_id )
      return;

    /* Fill up the buffer */
    if ( planner_manager_->swarm_trajs_buf_.size() <= id )
    {
      for ( size_t i=planner_manager_->swarm_trajs_buf_.size(); i<=id; i++ )
      {
        OneTrajDataOfSwarm blank;
        blank.drone_id = -1;
        planner_manager_->swarm_trajs_buf_.push_back(blank);
      }
    }

    /* Store data */
    Eigen::MatrixXd pos_pts(3, msg->pos_pts.size());
    Eigen::VectorXd knots(msg->knots.size());
    for (size_t j = 0; j < msg->knots.size(); ++j)
    {
      knots(j) = msg->knots[j];
    }
    for (size_t j = 0; j < msg->pos_pts.size(); ++j)
    {
      pos_pts(0, j) = msg->pos_pts[j].x;
      pos_pts(1, j) = msg->pos_pts[j].y;
      pos_pts(2, j) = msg->pos_pts[j].z;
    }

    planner_manager_->swarm_trajs_buf_[id].drone_id = id;

    if (msg->order % 2)
    {
      double cutback = (double)msg->order / 2 + 1.5;
      planner_manager_->swarm_trajs_buf_[id].duration_ = msg->knots[msg->knots.size() - ceil(cutback)];
    }
    else
    {
      double cutback = (double)msg->order / 2 + 1.5;
      planner_manager_->swarm_trajs_buf_[id].duration_ = (msg->knots[msg->knots.size() - floor(cutback)] + msg->knots[msg->knots.size() - ceil(cutback)]) / 2;
    }

    int size_pos = msg->pos_pts.size() - 1;

    Eigen::Vector3d start_p, end_p;
    start_p(0) = msg->pos_pts[0].x;
    start_p(1) = msg->pos_pts[0].y;
    start_p(2) = msg->pos_pts[0].z;    
    end_p(0) = msg->pos_pts[size_pos].x;
    end_p(1) = msg->pos_pts[size_pos].y;
    end_p(2) = msg->pos_pts[size_pos].z; 

    static int predict_int = 0;
    if( predict_int < 2 )
    {
      predict_int++;
      return;
    }

    if( (start_p - end_p).norm() < 0.5 )
    {
      changeFSMExecState(WAIT_TARGET, "PREDICT_CHECK");
      // ROS_WARN("Already for tracking!");
      return;
    }

    // cout << "Triggered!Predict!" << endl;

    // constexpr double time_step = 0.01;
    // double t_cur = 0;
    // Eigen::Vector3d p_cur = info->position_traj_.evaluateDeBoorT(t_cur);
    // for (double t = t_cur; t < info->duration_; t += time_step)
    // {
    //   bool occ = false;
    //   occ |= map->getInflateOccupancy(info->position_traj_.evaluateDeBoorT(t));
    // }

    for( size_t ii = 0; ii < msg->pos_pts.size(); ii++ )
    {
      // wxx
      if( planner_manager_->grid_map_->getInflateOccupancy( Eigen::Vector3d{pos_pts(0, ii), pos_pts(1, ii), pos_pts(2, ii)} ) == 1 )
      {
        ROS_ERROR("Predict are in the collision!");
        if( ii < 3 )
          ROS_ERROR("The first 3 points of predict are in the collision!");
        planner_manager_->reboundReplanForPredict(pos_pts, msg->knots[1] - msg->knots[0]);
        break;
      }
    }

    std::vector<Eigen::Vector3d> sample_list_predict;
    sample_list_predict.clear();

    for( size_t ii = 0; ii < msg->pos_pts.size(); ii++ )
    {
      sample_list_predict.push_back( Eigen::Vector3d{pos_pts(0, ii), pos_pts(1, ii), pos_pts(2, ii)} );
    }

    Eigen::Vector4d color(0, 1, 0.5, 1);
    visualization_->displaySphereList(new_predict_list_pub_, sample_list_predict, 0.8, color, 10);

    UniformBspline pos_traj(pos_pts, msg->order, msg->knots[1] - msg->knots[0]);
    pos_traj.setKnot(knots);
    planner_manager_->swarm_trajs_buf_[id].position_traj_ = pos_traj;

    planner_manager_->swarm_trajs_buf_[id].start_pos_ = planner_manager_->swarm_trajs_buf_[id].position_traj_.evaluateDeBoorT(0);

    planner_manager_->swarm_trajs_buf_[id].start_time_ = msg->start_time;
    
    receive_target_traj_ = true;


    predict_target_(0) = (msg->pos_pts[0].x * 10.0 + msg->pos_pts[size_pos].x * 10.0 ) / 20.0; 
    predict_target_(1) = (msg->pos_pts[0].y * 10.0 + msg->pos_pts[size_pos].y * 10.0 ) / 20.0; 
    predict_target_(2) = (msg->pos_pts[0].z * 10.0 + msg->pos_pts[size_pos].z * 10.0 ) / 20.0; 

    predict_vel_ = planner_manager_->swarm_trajs_buf_[id].position_traj_.getDerivative().evaluateDeBoorT(planner_manager_->swarm_trajs_buf_[id].duration_/2);
    
    // wxx
    if (exec_state_ == WAIT_TARGET)
      changeFSMExecState(GEN_NEW_TRAJ, "PREDICT_CHECK");
    else if (exec_state_ == EXEC_TRAJ)
      changeFSMExecState(REPLAN_TRAJ, "PREDICT_CHECK");

  }

  void EGOReplanFSM::swarmTrajsCallback(const traj_utils::MultiBsplinesPtr &msg)
  {
    
    multi_bspline_msgs_buf_.traj.clear();
    multi_bspline_msgs_buf_ = *msg;

    // cout << "\033[45;33mmulti_bspline_msgs_buf.drone_id_from=" << multi_bspline_msgs_buf_.drone_id_from << " multi_bspline_msgs_buf_.traj.size()=" << multi_bspline_msgs_buf_.traj.size() << "\033[0m" << endl;

    if ( !have_odom_ )
    {
      ROS_ERROR("swarmTrajsCallback(): no odom!, return.");
      return;
    }

    if ((int)msg->traj.size() != msg->drone_id_from + 1) // drone_id must start from 0
    {
      ROS_ERROR("Wrong trajectory size! msg->traj.size()=%d, msg->drone_id_from+1=%d", msg->traj.size(), msg->drone_id_from + 1);
      return;
    }

    if (msg->traj[0].order != 3) // only support B-spline order equals 3.
    {
      ROS_ERROR("Only support B-spline order equals 3.");
      return;
    }

    // Step 1. receive the trajectories
    planner_manager_->swarm_trajs_buf_.clear();
    planner_manager_->swarm_trajs_buf_.resize(msg->traj.size());

    for (size_t i = 0; i < msg->traj.size(); i++)
    {

      Eigen::Vector3d cp0(msg->traj[i].pos_pts[0].x, msg->traj[i].pos_pts[0].y, msg->traj[i].pos_pts[0].z);
      Eigen::Vector3d cp1(msg->traj[i].pos_pts[1].x, msg->traj[i].pos_pts[1].y, msg->traj[i].pos_pts[1].z);
      Eigen::Vector3d cp2(msg->traj[i].pos_pts[2].x, msg->traj[i].pos_pts[2].y, msg->traj[i].pos_pts[2].z);
      Eigen::Vector3d swarm_start_pt = (cp0+4*cp1+cp2) / 6;
      if ( (swarm_start_pt - odom_pos_).norm() > planning_horizen_*4.0f/3.0f )
      {
        planner_manager_->swarm_trajs_buf_[i].drone_id = -1;
        continue;
      }

      Eigen::MatrixXd pos_pts(3, msg->traj[i].pos_pts.size());
      Eigen::VectorXd knots(msg->traj[i].knots.size());
      for (size_t j = 0; j < msg->traj[i].knots.size(); ++j)
      {
        knots(j) = msg->traj[i].knots[j];
      }
      for (size_t j = 0; j < msg->traj[i].pos_pts.size(); ++j)
      {
        pos_pts(0, j) = msg->traj[i].pos_pts[j].x;
        pos_pts(1, j) = msg->traj[i].pos_pts[j].y;
        pos_pts(2, j) = msg->traj[i].pos_pts[j].z;
      }

      planner_manager_->swarm_trajs_buf_[i].drone_id = i;

      if (msg->traj[i].order % 2)
      {
        double cutback = (double)msg->traj[i].order / 2 + 1.5;
        planner_manager_->swarm_trajs_buf_[i].duration_ = msg->traj[i].knots[msg->traj[i].knots.size() - ceil(cutback)];
      }
      else
      {
        double cutback = (double)msg->traj[i].order / 2 + 1.5;
        planner_manager_->swarm_trajs_buf_[i].duration_ = (msg->traj[i].knots[msg->traj[i].knots.size() - floor(cutback)] + msg->traj[i].knots[msg->traj[i].knots.size() - ceil(cutback)]) / 2;
      }

      // planner_manager_->swarm_trajs_buf_[i].position_traj_ =
      UniformBspline pos_traj(pos_pts, msg->traj[i].order, msg->traj[i].knots[1] - msg->traj[i].knots[0]);
      pos_traj.setKnot(knots);
      planner_manager_->swarm_trajs_buf_[i].position_traj_ = pos_traj;

      planner_manager_->swarm_trajs_buf_[i].start_pos_ = planner_manager_->swarm_trajs_buf_[i].position_traj_.evaluateDeBoorT(0);

      planner_manager_->swarm_trajs_buf_[i].start_time_ = msg->traj[i].start_time;
    }

    have_recv_pre_agent_ = true;
  }

  void EGOReplanFSM::changeFSMExecState(FSM_EXEC_STATE new_state, string pos_call)
  {

    if (new_state == exec_state_)
      continously_called_times_++;
    else
      continously_called_times_ = 1;

    static string state_str[8] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP", "SEQUENTIAL_START"};
    int pre_s = int(exec_state_);
    exec_state_ = new_state;
    cout << "[" + pos_call + "]: from " + state_str[pre_s] + " to " + state_str[int(new_state)] << endl;
  }

  std::pair<int, EGOReplanFSM::FSM_EXEC_STATE> EGOReplanFSM::timesOfConsecutiveStateCalls()
  {
    return std::pair<int, FSM_EXEC_STATE>(continously_called_times_, exec_state_);
  }

  void EGOReplanFSM::printFSMExecState()
  {
    static string state_str[8] = {"INIT", "WAIT_TARGET", "GEN_NEW_TRAJ", "REPLAN_TRAJ", "EXEC_TRAJ", "EMERGENCY_STOP", "SEQUENTIAL_START"};

    cout << "[FSM]: state: " + state_str[int(exec_state_)] << endl;
  }

  void EGOReplanFSM::execFSMCallback(const ros::TimerEvent &e)
  {
    exec_timer_.stop();  // To avoid blockage

    static int fsm_num = 0;

    // // wxx : debug for static 
    // static int need_replan = 0;
    // static int lalalala = 0;
    // if( fsm_num < 500 )
    //   fsm_num++;
    // else
    // {
    //   if( planner_manager_->pp_.drone_id == 1 )
    //   {
    //     if( have_target_ )
    //     {
    //       have_target_ = false;
    //       bool success = planFromGlobalTraj(10); // fuck

    //       if (success)
    //       {
    //         publishSwarmTrajs(false);
    //         ROS_WARN(" Drone 1 Plan Success!!!");
    //       }
    //       else
    //       {
    //         ROS_WARN(" Drone 1 Plan Fail!!! ");
    //       }
    //     }
    //   }
    //   else
    //   {
    //     if( receive_target_traj_ )
    //     {
    //       receive_target_traj_ = false;
    //       bool success = planFromGlobalTraj(10); // fuck
    //       if (success)
    //       {
    //         publishSwarmTrajs(false);
    //         ROS_WARN("Drone 0 Plan Success!!!");
    //       }
    //       else
    //       {
    //         ROS_WARN(" Drone 0 Plan Fail!!! ");
    //       }
    //     }
    //   }
    // }
    // return;

    fsm_num++;
    if (fsm_num == 100)
    {
      printFSMExecState();
      if (!have_odom_)
        cout << "no odom." << endl;
      if (!have_target_)
        cout << "wait for goal or trigger." << endl;
      fsm_num = 0;
    }

    switch (exec_state_)
    {
    case INIT:
    {
      if (!have_odom_ )
      {
        goto force_return;
        // return;
      }
      changeFSMExecState(WAIT_TARGET, "FSM");
      break;
    }

    case WAIT_TARGET:
    {
      if (!have_target_)
        goto force_return;
        // return;
      else
      {
        // if ( planner_manager_->pp_.drone_id <= 0 )
        // {
        //   changeFSMExecState(GEN_NEW_TRAJ, "FSM");
        // }
        // else
        // {
        changeFSMExecState(SEQUENTIAL_START, "FSM");
        // }
      }
      break;
    }

    case SEQUENTIAL_START: // for swarm
    {
      // cout << "id=" << planner_manager_->pp_.drone_id << " have_recv_pre_agent_=" << have_recv_pre_agent_ << endl;
      if (  planner_manager_->pp_.drone_id <= 0 || (planner_manager_->pp_.drone_id >= 1  && have_recv_pre_agent_) )
      {
        if (have_odom_ && have_target_)
        {
          bool success = planFromGlobalTraj(10); // zx-todo
          if (success)
          {
            changeFSMExecState(EXEC_TRAJ, "FSM");
            
            publishSwarmTrajs(true);
          }
          else
          {
            ROS_ERROR("Failed to generate the first trajectory!!!");
            // todo
          }
        }
        else
        {
          ROS_ERROR("No odom or no target! have_odom_=%d, have_target_=%d", have_odom_, have_target_);
        }
      }

      break;
    }

    case GEN_NEW_TRAJ:
    {

      // Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
      // start_yaw_(0)         = atan2(rot_x(1), rot_x(0));
      // start_yaw_(1) = start_yaw_(2) = 0.0;

      bool success = planFromGlobalTraj(10); // zx-todo
      if (success)
      {
        changeFSMExecState(EXEC_TRAJ, "FSM");
        flag_escape_emergency_ = true;
        publishSwarmTrajs(false);
      }
      else
      {
        changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }
      break;
    }

    case REPLAN_TRAJ:
    {

      if (planFromCurrentTraj(1))
      {
        changeFSMExecState(EXEC_TRAJ, "FSM");
        publishSwarmTrajs(false);
      }
      else
      {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }

      break;
    }

    case EXEC_TRAJ:
    {
      /* determine if need to replan */
      LocalTrajData *info = &planner_manager_->local_data_;
      ros::Time time_now = ros::Time::now();
      double t_cur = (time_now - info->start_time_).toSec();
      t_cur = min(info->duration_, t_cur);

      Eigen::Vector3d pos = info->position_traj_.evaluateDeBoorT(t_cur);

      /* && (end_pt_ - pos).norm() < 0.5 */
      if ((local_target_pt_ - end_pt_).norm() < 1e-3) // close to the global target
      {
        if (t_cur > info->duration_ - 1e-2)
        {
          have_target_ = false;

          changeFSMExecState(WAIT_TARGET, "FSM");
          goto force_return;
          // return;
        }
        else if ((end_pt_ - pos).norm() > no_replan_thresh_ && t_cur > replan_thresh_)
        {
          changeFSMExecState(REPLAN_TRAJ, "FSM");
        }
      }
      else if (t_cur > replan_thresh_)
      {
        changeFSMExecState(REPLAN_TRAJ, "FSM");
      }

      break;
    }

    case EMERGENCY_STOP:
    {

      if (flag_escape_emergency_) // Avoiding repeated calls
      {
        callEmergencyStop(odom_pos_);
      }
      else
      {
        if (enable_fail_safe_ && odom_vel_.norm() < 0.1)
          changeFSMExecState(GEN_NEW_TRAJ, "FSM");
      }

      flag_escape_emergency_ = false;
      break;
    }
    }

    data_disp_.header.stamp = ros::Time::now();
    data_disp_pub_.publish(data_disp_);

    force_return:;
    exec_timer_.start();
  }

  bool EGOReplanFSM::planFromGlobalTraj(const int trial_times /*=1*/) //zx-todo
  {
    start_pt_ = odom_pos_;
    start_vel_ = odom_vel_;
    start_acc_.setZero();

    Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
    start_yaw_(0)         = atan2(rot_x(1), rot_x(0));
    start_yaw_(1) = start_yaw_(2) = 0.0;

    bool flag_random_poly_init;
    if (timesOfConsecutiveStateCalls().first == 1)
      flag_random_poly_init = false;
    else
      flag_random_poly_init = true;

    for (int i = 0; i < trial_times; i++)
    {
      if (callReboundReplan(true, flag_random_poly_init))
      {
        return true;
      }
    }
    return false;
  }

  bool EGOReplanFSM::planFromCurrentTraj(const int trial_times /*=1*/)
  {

    LocalTrajData *info = &planner_manager_->local_data_;
    ros::Time time_now = ros::Time::now();
    double t_cur = (time_now - info->start_time_).toSec();

    //cout << "info->velocity_traj_=" << info->velocity_traj_.get_control_points() << endl;

    start_pt_ = info->position_traj_.evaluateDeBoorT(t_cur);
    start_vel_ = info->velocity_traj_.evaluateDeBoorT(t_cur);
    start_acc_ = info->acceleration_traj_.evaluateDeBoorT(t_cur);


    if( planner_manager_->pp_.drone_id == 0 )
    {
      start_yaw_(0) = info->yaw_traj_.evaluateDeBoorT(t_cur)[0];
      start_yaw_(1) = info->yawdot_traj_.evaluateDeBoorT(t_cur)[0];
      start_yaw_(2) = 0;
    }
    else
    {
      Eigen::Vector3d rot_x = odom_orient_.toRotationMatrix().block(0, 0, 3, 1);
      start_yaw_(0)         = atan2(rot_x(1), rot_x(0));
      start_yaw_(1) = start_yaw_(2) = 0.0;
    }


    bool success = callReboundReplan(false, false);

    if (!success)
    {
      success = callReboundReplan(true, false);
      //changeFSMExecState(EXEC_TRAJ, "FSM");
      if (!success)
      {
        for (int i = 0; i < trial_times; i++)
        {
          success = callReboundReplan(true, true);
          if (success)
            break;
        }
        if (!success)
        {
          return false;
        }
      }
    }

    return true;
  }

  void EGOReplanFSM::checkCollisionCallback(const ros::TimerEvent &e)
  {

    LocalTrajData *info = &planner_manager_->local_data_;
    auto map = planner_manager_->grid_map_;

    if (exec_state_ == WAIT_TARGET || info->start_time_.toSec() < 1e-5)
      return;

    /* ---------- check lost of depth ---------- */
    if ( map->getOdomDepthTimeout() )
    {
      ROS_ERROR("Depth Lost! EMERGENCY_STOP");
      enable_fail_safe_ = false;
      changeFSMExecState(EMERGENCY_STOP, "SAFETY");
    }

    /* ---------- check trajectory ---------- */
    constexpr double time_step = 0.01;
    double t_cur = (ros::Time::now() - info->start_time_).toSec();
    Eigen::Vector3d p_cur = info->position_traj_.evaluateDeBoorT(t_cur);
    const double CLEARANCE = 1.0 * planner_manager_->getSwarmClearance();
    double t_cur_global = ros::Time::now().toSec();
    double t_2_3 = info->duration_ * 2 / 3;
    for (double t = t_cur; t < info->duration_; t += time_step)
    {
      if (t_cur < t_2_3 && t >= t_2_3) // If t_cur < t_2_3, only the first 2/3 partition of the trajectory is considered valid and will get checked.
        break;

      bool occ = false;
      occ |= map->getInflateOccupancy(info->position_traj_.evaluateDeBoorT(t));

      for (size_t id = 0; id < planner_manager_->swarm_trajs_buf_.size(); id++)
      {
        if ( (planner_manager_->swarm_trajs_buf_.at(id).drone_id != (int)id) || (planner_manager_->swarm_trajs_buf_.at(id).drone_id == planner_manager_->pp_.drone_id) )
        {
          continue;
        }

        double t_X = t_cur_global - planner_manager_->swarm_trajs_buf_.at(id).start_time_.toSec();
        Eigen::Vector3d swarm_pridicted = planner_manager_->swarm_trajs_buf_.at(id).position_traj_.evaluateDeBoorT(t_X);
        double dist = (p_cur - swarm_pridicted).norm();

        if ( dist < CLEARANCE )
        {
          occ = true;
          break;
        }
      }

      if (occ)
      {

        if (planFromCurrentTraj()) // Make a chance
        {
          changeFSMExecState(EXEC_TRAJ, "SAFETY");
          publishSwarmTrajs(false);
          return;
        }
        else
        {
          if (t - t_cur < emergency_time_) // 0.8s of emergency time
          {
            ROS_WARN("Suddenly discovered obstacles. emergency stop! time=%f", t - t_cur);
            changeFSMExecState(EMERGENCY_STOP, "SAFETY");
          }
          else
          {
            //ROS_WARN("current traj in collision, replan.");
            changeFSMExecState(REPLAN_TRAJ, "SAFETY");
          }
          return;
        }
        break;
      }
    }
  }

  bool EGOReplanFSM::callReboundReplan(bool flag_use_poly_init, bool flag_randomPolyTraj)
  {
    getLocalTarget();

    static int count_just_see = 0;
    printf("\033[47;30m\n[drone replan %d start]==============================================\033[0m\n", count_just_see++);

    bool kinodynamic_success =
        planner_manager_->kinodynamicReplan(start_pt_, start_vel_, start_acc_, local_target_pt_, local_target_vel_);

    if(!kinodynamic_success)
        kinodynamic_success = planner_manager_->reboundReplan(start_pt_, start_vel_, start_acc_, local_target_pt_, local_target_vel_, (have_new_target_ || flag_use_poly_init), flag_randomPolyTraj);
    
    have_new_target_ = false;

    cout << "kinodynamic A*=" << kinodynamic_success << endl;

    bool yaw_plan_success = false;
    if (kinodynamic_success)
    {
      yaw_plan_success = planner_manager_->reboundReplanWithYaw(start_yaw_, local_target_vel_);
      cout << "yaw_plan_success=" << yaw_plan_success << endl;
    }

    if( yaw_plan_success )
    {
      /****************************************************/
      /********************* pub traj *********************/
      /****************************************************/
      auto info = &planner_manager_->local_data_;
      traj_utils::Bspline bspline;
      bspline.order = 3;
      bspline.start_time = info->start_time_;
      bspline.traj_id = info->traj_id_;

      Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
      bspline.pos_pts.reserve(pos_pts.cols());
      for (int i = 0; i < pos_pts.cols(); ++i)
      {
        geometry_msgs::Point pt;
        pt.x = pos_pts(0, i);
        pt.y = pos_pts(1, i);
        pt.z = pos_pts(2, i);
        bspline.pos_pts.push_back(pt);
      }

      Eigen::VectorXd knots = info->position_traj_.getKnot();
      // cout << knots.transpose() << endl;
      bspline.knots.reserve(knots.rows());
      for (int i = 0; i < knots.rows(); ++i)
      {
        bspline.knots.push_back(knots(i));
      }

      // yaw 的轨迹
      Eigen::MatrixXd yaw_pts = info->yaw_traj_.getControlPoint();
      bspline.yaw_pts.reserve(yaw_pts.cols());
      for (int i = 0; i < yaw_pts.cols(); ++i)
      {
        double yaw;
        yaw = yaw_pts(0,i);
        bspline.yaw_pts.push_back(yaw);
      }
      bspline.have_yaw = true;

      bspline_pub_.publish(bspline);


      /****************************************************/
      /***********************可视化************************/
      /****************************************************/

      // show control points
      vector<Eigen::Vector3d> cpt_list;
      for (int i = 0; i < pos_pts.cols(); ++i)
      {
        Eigen::Vector3d pt;
        pt(0) = pos_pts(0, i);
        pt(1) = pos_pts(1, i);
        pt(2) = pos_pts(2, i);
        cpt_list.push_back(pt);
      }
      Eigen::Vector4d color0(0, 1, 1, 1);
      visualization_->displaySphereList(cpt_list_pub_, cpt_list, 0.3, color0, 100);

      // show pos traj
      UniformBspline pos_traj(pos_pts, 3, 0.1);
      pos_traj.setKnot(knots);
      double duration = info->duration_;
      vector<Eigen::Vector3d> pos_list;
      Eigen::Vector3d pos(Eigen::Vector3d::Zero()), yaw_pos(Eigen::Vector3d::Zero()), cpt_pos(Eigen::Vector3d::Zero());
      double display_dt = 0.1;
      for( double t = 0; t < duration; t = t + display_dt )
      {
        double t_now = t ;
        pos = pos_traj.evaluateDeBoorT(t_now);
        pos_list.push_back(pos);
      }
      Eigen::Vector4d color1(1, 0, 0, 1);
      visualization_->displayMarkerList(pos_list_pub_, pos_list, 0.15, color1, 100, false);

      // show yaw
      UniformBspline yaw_traj(yaw_pts, 3, 0.1);
      yaw_traj.setKnot(knots);
      vector<Eigen::Vector3d> yaw_list;
      display_dt = duration / pos_pts.cols();
      for( double t = 0; t < duration; t = t + display_dt )
      {
        double t_now = t ;
        pos = pos_traj.evaluateDeBoorT(t_now);
        yaw_list.push_back(pos);

        double yaw = yaw_traj.evaluateDeBoorT(t_now)[0];
        yaw_pos = pos + Eigen::Vector3d{ cos(yaw), sin(yaw), 0};
        yaw_list.push_back(yaw_pos);
      }
      Eigen::Vector4d color2(0, 1, 0, 0.2);
      visualization_->displayArrowList (yaw_list_pub_, yaw_list, 0.1 , color2, 2000);
      

      // show some something for debug
      // eg. visibility; gradient and so on
      std::vector<Eigen::Vector3d> vector_lines, gradient_lines, pk_gradient_lines;
      cpt_list.clear();
      vector_lines.clear();
      gradient_lines.clear();
      for (int ii = 0; ii < planner_manager_->bspline_optimizer_->visibility_index.size(); ++ii)
      {
        int i = planner_manager_->bspline_optimizer_->visibility_index[ii];
        Eigen::Vector3d pt;
        pt(0) = pos_pts(0, i);
        pt(1) = pos_pts(1, i);
        pt(2) = pos_pts(2, i);
        Eigen::Vector3d pt_target;

        cpt_list.push_back(pt);
        pt_target = planner_manager_->bspline_optimizer_->visibility_target_point[ii];

        vector_lines.push_back(pt);
        vector_lines.push_back(pt_target);

        pt_target = planner_manager_->bspline_optimizer_->visibility_gradient[ii];
        pt_target = pt + pt_target;
        gradient_lines.push_back(pt);
        gradient_lines.push_back(pt_target);
      }
      Eigen::Vector4d color(0.5, 0, 0.8, 0.8);
      Eigen::Vector4d color_gradient(0, 0.5, 0.8, 0.8);
      visualization_->displayArrowList (attract_score_list_pub_, vector_lines, 0.05 , color, 1000);
      visualization_->displayArrowList (gradient_list_pub_, gradient_lines, 0.05 , color_gradient, 2000);
      visualization_->displaySphereList(debug_cpt_list_pub_, cpt_list, 0.3, color, 102);

      // 梯度的可视化 
      pk_gradient_lines.clear();
      for( int ii = 0; ii < planner_manager_->bspline_optimizer_->vis_pk_grad.size(); ii++ )
      {
        pk_gradient_lines.push_back(planner_manager_->bspline_optimizer_->vis_pk[ii]);
        pk_gradient_lines.push_back(planner_manager_->bspline_optimizer_->vis_pk[ii] + planner_manager_->bspline_optimizer_->vis_pk_grad[ii]);
        pk_gradient_lines.push_back(planner_manager_->bspline_optimizer_->vis_pk[ii]);
        pk_gradient_lines.push_back(planner_manager_->bspline_optimizer_->vis_pk[ii] + planner_manager_->bspline_optimizer_->vis_pk_grad_real[ii]);
      }
      Eigen::Vector4d color_pk_gradient(0.2, 0.5, 0.2, 0.8);
      visualization_->displayArrowList (pk_grad_list_pub_, pk_gradient_lines, 0.02 , color_pk_gradient, 3000);
    }

    return yaw_plan_success;
  }

  void EGOReplanFSM::publishSwarmTrajs(bool startup_pub)
  {
    auto info = &planner_manager_->local_data_;

    traj_utils::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.drone_id = planner_manager_->pp_.drone_id;
    bspline.traj_id = info->traj_id_;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    bspline.pos_pts.reserve(pos_pts.cols());
    for (int i = 0; i < pos_pts.cols(); ++i)
    {
      geometry_msgs::Point pt;
      pt.x = pos_pts(0, i);
      pt.y = pos_pts(1, i);
      pt.z = pos_pts(2, i);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();
    // cout << knots.transpose() << endl;
    bspline.knots.reserve(knots.rows());
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }

    if ( startup_pub )
    {
      multi_bspline_msgs_buf_.drone_id_from = planner_manager_->pp_.drone_id; // zx-todo
      if ((int)multi_bspline_msgs_buf_.traj.size() == planner_manager_->pp_.drone_id + 1)
      {
        multi_bspline_msgs_buf_.traj.back() = bspline;
      }
      else if ((int)multi_bspline_msgs_buf_.traj.size() == planner_manager_->pp_.drone_id)
      {
        multi_bspline_msgs_buf_.traj.push_back(bspline);
      }
      else
      {
        ROS_ERROR("Wrong traj nums and drone_id pair!!! traj.size()=%d, drone_id=%d", multi_bspline_msgs_buf_.traj.size(), planner_manager_->pp_.drone_id);
        // return plan_and_refine_success;
      }
      swarm_trajs_pub_.publish(multi_bspline_msgs_buf_);
    }

    // broadcast_bspline_pub_.publish(bspline);
  }

  bool EGOReplanFSM::callEmergencyStop(Eigen::Vector3d stop_pos)
  {

    planner_manager_->EmergencyStop(stop_pos);

    auto info = &planner_manager_->local_data_;

    /* publish traj */
    traj_utils::Bspline bspline;
    bspline.order = 3;
    bspline.start_time = info->start_time_;
    bspline.traj_id = info->traj_id_;

    Eigen::MatrixXd pos_pts = info->position_traj_.getControlPoint();
    bspline.pos_pts.reserve(pos_pts.cols());
    for (int i = 0; i < pos_pts.cols(); ++i)
    {
      geometry_msgs::Point pt;
      pt.x = pos_pts(0, i);
      pt.y = pos_pts(1, i);
      pt.z = pos_pts(2, i);
      bspline.pos_pts.push_back(pt);
    }

    Eigen::VectorXd knots = info->position_traj_.getKnot();
    bspline.knots.reserve(knots.rows());
    for (int i = 0; i < knots.rows(); ++i)
    {
      bspline.knots.push_back(knots(i));
    }

    bspline_pub_.publish(bspline);

    return true;
  }

  void EGOReplanFSM::getLocalTarget()
  {
    if( planner_manager_->pp_.drone_id == 0 )
    {
      local_target_pt_ = predict_target_;
      local_target_vel_ = predict_vel_;
      // local_target_vel_ = Eigen::Vector3d::Zero();
      std::cout << "Get Local Target!" << std::endl;
      return;
    }

    double t;

    double t_step = planning_horizen_ / 20 / planner_manager_->pp_.max_vel_;
    double dist_min = 9999, dist_min_t = 0.0;
    for (t = planner_manager_->global_data_.last_progress_time_; t < planner_manager_->global_data_.global_duration_; t += t_step)
    {
      Eigen::Vector3d pos_t = planner_manager_->global_data_.getPosition(t);
      double dist = (pos_t - start_pt_).norm();

      if (t < planner_manager_->global_data_.last_progress_time_ + 1e-5 && dist > planning_horizen_)
      {
        // todo
        ROS_ERROR("last_progress_time_ ERROR, TODO!");
        ROS_ERROR("last_progress_time_ ERROR, TODO!");
        ROS_ERROR("last_progress_time_ ERROR, TODO!");
        ROS_ERROR("last_progress_time_ ERROR, TODO!");
        ROS_ERROR("last_progress_time_ ERROR, TODO!");
        cout << "dist=" << dist << endl;
        cout << "planner_manager_->global_data_.last_progress_time_=" << planner_manager_->global_data_.last_progress_time_ << endl;
        return;
      }
      if (dist < dist_min)
      {
        dist_min = dist;
        dist_min_t = t;
      }
      if (dist >= planning_horizen_)
      {
        local_target_pt_ = pos_t;
        planner_manager_->global_data_.last_progress_time_ = dist_min_t;
        break;
      }
    }
    if (t > planner_manager_->global_data_.global_duration_) // Last global point
    {
      // planner_manager_->grid_map_;
      local_target_pt_ = end_pt_;
    }

    if ((end_pt_ - local_target_pt_).norm() < (planner_manager_->pp_.max_vel_ * planner_manager_->pp_.max_vel_) / (2 * planner_manager_->pp_.max_acc_))
    {
      // local_target_vel_ = (end_pt_ - init_pt_).normalized() * planner_manager_->pp_.max_vel_ * (( end_pt_ - local_target_pt_ ).norm() / ((planner_manager_->pp_.max_vel_*planner_manager_->pp_.max_vel_)/(2*planner_manager_->pp_.max_acc_)));
      // cout << "A" << endl;
      local_target_vel_ = Eigen::Vector3d::Zero();
    }
    else
    {
      local_target_vel_ = planner_manager_->global_data_.getVelocity(t);
      // cout << "AA" << endl;
    }
  }

} // namespace ego_planner
