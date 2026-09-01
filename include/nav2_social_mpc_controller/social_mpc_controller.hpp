// Copyright (c) 2022 SRL -Service Robotics Lab, Pablo de Olavide University
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_SOCIAL_MPC_CONTROLLER__MPC_CONTROLLER_HPP_
#define NAV2_SOCIAL_MPC_CONTROLLER__MPC_CONTROLLER_HPP_

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose2_d.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "nav2_core/controller.hpp"
#include "nav2_social_mpc_controller/obstacle_distance_interface.hpp"
#include "nav2_social_mpc_controller/optimizer.hpp"
#include "nav2_social_mpc_controller/path_trajectorizer.hpp"
#include "nav2_social_mpc_controller/people_interface.hpp"
#include "nav2_util/odometry_utils.hpp"
#include "obstacle_distance_msgs/msg/obstacle_distance.hpp"
#include "people_msgs/msg/people.hpp"
#include "std_msgs/msg/bool.hpp"
#include "pluginlib/class_list_macros.hpp"
#include "pluginlib/class_loader.hpp"
#include "rclcpp/rclcpp.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_util/robot_utils.hpp"
#include "nav2_social_mpc_controller/tools/path_handler.hpp"
#include "nav2_social_mpc_controller/tools/type_definitions.hpp"

namespace nav2_social_mpc_controller
{

/**
 * @class nav2_social_mpc_controller::SocialMPCController
 * @brief social mpc controller plugin
 */
class SocialMPCController : public nav2_core::Controller
{
public:
  /**
   * @brief Constructor for
   * nav2_social_mpc_controller::SocialMPCController
   */
  SocialMPCController() = default;

  /**
   * @brief Destrructor for
   * nav2_social_mpc_controller::SocialMPCController
   */
  ~SocialMPCController() override = default;

  /**
   * @brief Configure controller state machine
   * @param parent WeakPtr to node
   * @param name Name of node
   * @param tf TF buffer
   * @param costmap_ros Costmap2DROS object of environment
   */
  void configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent, std::string name,
                 std::shared_ptr<tf2_ros::Buffer> tf,
                 std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros) override;

  /**
   * @brief Cleanup controller state machine
   */
  void cleanup() override;

  /**
   * @brief Activate controller state machine
   */
  void activate() override;

  /**
   * @brief Deactivate controller state machine
   */
  void deactivate() override;

  /**
   * @brief Compute the best command given the current pose and velocity, with
   * possible debug information
   *
   * Same as above computeVelocityCommands, but with debug results.
   * If the results pointer is not null, additional information about the twists
   * evaluated will be in results after the call.
   *
   * @param pose      Current robot pose
   * @param velocity  Current robot velocity
   * @param results   Output param, if not NULL, will be filled in with full
   * evaluation results
   * @return          Best command
   */
  geometry_msgs::msg::TwistStamped computeVelocityCommands(const geometry_msgs::msg::PoseStamped& pose,
                                                           const geometry_msgs::msg::Twist& velocity,
                                                           nav2_core::GoalChecker* goal_checker) override;
  void setSpeedLimit(const double& speed_limit, const bool& percentage) override;

  /**
   * @brief nav2_core setPlan - Sets the global plan
   * @param path The global plan
   */
  void setPlan(const nav_msgs::msg::Path& path) override;

  void publish_people_traj(const AgentsTrajectories& people, const std_msgs::msg::Header& header);

protected:
  // /**
  //  * @brief Transforms global plan into same frame as pose, clips far away poses
  //  * and possibly prunes passed poses
  //  * @param pose pose to transform
  //  * @return Path in new frame
  //  */
  // nav_msgs::msg::Path transformGlobalPlan(const geometry_msgs::msg::PoseStamped& pose);

  /**
   * @brief Transform a pose to another frame.
   * @param frame Frame ID to transform to
   * @param in_pose Pose input to transform
   * @param out_pose transformed output
   * @return bool if successful
   */
  bool transformPose(const std::string frame, const geometry_msgs::msg::PoseStamped& in_pose,
                     geometry_msgs::msg::PoseStamped& out_pose) const;

  /**
   * @brief
   *
   * @param frame
   * @param in_point
   * @param out_point
   * @return true
   * @return false
   */
  bool transformPoint(const std::string frame, const geometry_msgs::msg::PointStamped& in_point,
                      geometry_msgs::msg::PointStamped& out_point) const;

  std::shared_ptr<tf2_ros::Buffer> tf_;
  std::string plugin_name_;
  std::unique_ptr<PathTrajectorizer> trajectorizer_;
  std::unique_ptr<Optimizer> optimizer_;
  std::unique_ptr<PeopleInterface> people_interface_;
  std::unique_ptr<ObstacleDistInterface> obsdist_interface_;
  OptimizerParams optimizer_params_;
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros_;
  nav2_costmap_2d::Costmap2D* costmap_;
  rclcpp::Logger logger_{ rclcpp::get_logger("SocialMPCController") };

  double speed_limit;
  bool percentage;
  double desired_linear_vel_;
  // Speeds as configured, before setSpeedLimit() scaling. Limits apply against THESE, not the live
  // values, so repeated calls cannot ratchet toward zero and clearing one restores the config.
  double base_desired_linear_vel_{0.0};
  double base_max_linear_vel_{0.0};
  double lookahead_dist_;
  double rotate_to_heading_angular_vel_;
  double max_lookahead_dist_;
  double min_lookahead_dist_;
  double lookahead_time_;
  bool use_velocity_scaled_lookahead_dist_;
  tf2::Duration transform_tolerance_;
  bool use_approach_vel_scaling_;
  double min_approach_linear_velocity_;
  double control_duration_;
  double max_allowed_time_to_collision_;
  bool use_regulated_linear_velocity_scaling_;
  bool use_cost_regulated_linear_velocity_scaling_;
  double cost_scaling_dist_;
  double cost_scaling_gain_;
  double inflation_cost_scaling_factor_;
  double regulated_linear_scaling_min_radius_;
  double regulated_linear_scaling_min_speed_;
  bool use_rotate_to_heading_;
  double max_angular_accel_;
  double rotate_to_heading_min_angle_;
  double goal_dist_tol_;
  double fov_angle_;
  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Path>> local_path_pub_;

  std::shared_ptr<rclcpp_lifecycle::LifecyclePublisher<visualization_msgs::msg::MarkerArray>> people_traj_pub_;
  std::unique_ptr<mpc::PathHandler> path_handler_;

  // Silence motion/STOP diagnostics while the sim is paused (social_yielding freeze):
  // during a freeze the robot is deliberately halted, so "STOPPED" logs are noise.
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr sim_running_sub_;
  bool sim_running_{true};

  // --- Command-chain taps, merged into the optimizer's [MOTION-DIAG] line ------------------
  // A cmd_vx/meas_vx gap says the robot is not flying what the MPC solved, but not WHICH stage
  // lost it. Chain: solved -> cmd_vel_nav -> smoother -> cmd_vel_smoothed -> collision_monitor
  // -> cmd_vel -> base. Tapping each stage lets [MOTION-DIAG] name the culprit.
  // See docs/实现细节.md for the topology and why the remap exists.
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_nav_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_smoothed_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_out_sub_;
  double last_cmd_nav_vx_{std::numeric_limits<double>::quiet_NaN()};
  double last_cmd_smoothed_vx_{std::numeric_limits<double>::quiet_NaN()};
  double last_cmd_out_vx_{std::numeric_limits<double>::quiet_NaN()};
  // Arrival time of the last message on each tap. STEADY clock, not sim: it measures whether the
  // subscription is alive, which must stay meaningful while the sim clock is frozen by a pause.
  std::shared_ptr<rclcpp::Clock> chain_clock_;
  rclcpp::Time last_cmd_nav_stamp_;
  rclcpp::Time last_cmd_smoothed_stamp_;
  rclcpp::Time last_cmd_out_stamp_;

  // --- Social-profile dynamic parameters -------------------------------------------------
  // Lets a scenario retune pedestrian weights at RUNTIME (profiles pushed via /set_parameters at
  // reset). Weights are otherwise read only at configure(), so a runtime push would be ignored.
  // Runs on controller_server's (single-threaded) executor, serialized with
  // computeVelocityCommands, so re-initializing the optimizer here cannot race an optimize().
  rcl_interfaces::msg::SetParametersResult dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters);
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr dyn_params_handler_;
};

}  // namespace nav2_social_mpc_controller

#endif  // NAV2_SOCIAL_MPC_CONTROLLER__MPC_CONTROLLER_HPP_
