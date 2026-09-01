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

#include "nav2_social_mpc_controller/social_mpc_controller.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#include "angles/angles.h"
#include "nav2_core/planner_exceptions.hpp"
#include "nav2_util/geometry_utils.hpp"
#include "nav2_util/node_utils.hpp"
#include "pluginlib/class_list_macros.hpp"

using nav2_util::declare_parameter_if_not_declared;
using nav2_util::geometry_utils::euclidean_distance;
using std::abs;
using std::hypot;
using std::max;
using std::min;
using namespace nav2_costmap_2d;  // NOLINT

double clamp(double value, double min, double max)
{
  if (value < min)
    return min;
  if (value > max)
    return max;
  return value;
}

namespace nav2_social_mpc_controller
{

void SocialMPCController::configure(const rclcpp_lifecycle::LifecycleNode::WeakPtr& parent, std::string name,
                                    std::shared_ptr<tf2_ros::Buffer> tf,
                                    std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros)
{
  auto node = parent.lock();
  costmap_ros_ = costmap_ros;
  costmap_ = costmap_ros_->getCostmap();
  tf_ = tf;
  plugin_name_ = name;
  logger_ = node->get_logger();
  double transform_tolerance;
  declare_parameter_if_not_declared(node, plugin_name_ + ".desired_linear_vel", rclcpp::ParameterValue(0.5));
  declare_parameter_if_not_declared(node, plugin_name_ + ".fov_angle", rclcpp::ParameterValue(M_PI / 4));
  declare_parameter_if_not_declared(node, plugin_name_ + ".transform_tolerance", rclcpp::ParameterValue(0.1));
  node->get_parameter(plugin_name_ + ".desired_linear_vel", desired_linear_vel_);
  node->get_parameter(plugin_name_ + ".transform_tolerance", transform_tolerance);
  transform_tolerance_ = tf2::durationFromSec(transform_tolerance);
  node->get_parameter(plugin_name_ + ".fov_angle", fov_angle_);
  // Create the trajectorizer
  trajectorizer_ = std::make_unique<PathTrajectorizer>();
  trajectorizer_->configure(node, name, tf_);
  optimizer_ = std::make_unique<Optimizer>();
  optimizer_params_.get(node.get(), name);
  optimizer_->initialize(optimizer_params_);

  // HORIZON ALIGNMENT, second half. OptimizerParams::get() clamped its own max_time to
  // control_horizon*time_step and warned if the config disagreed, but the trajectorizer derived
  // max_steps_ from the raw config value back in its configure() above, so it needs the corrected
  // value pushed in. Leaving the two out of step is what made the robot cruise ~25% over its
  // desired speed: DistanceCost aims every horizon step at the reference's final point, so a
  // reference longer than the horizon puts that point out of reach at the cruise speed.
  trajectorizer_->setMaxTime(optimizer_params_.max_time);

  // Snapshot the configured speeds for setSpeedLimit() to scale against.
  base_desired_linear_vel_ = optimizer_params_.desired_linear_vel_;
  base_max_linear_vel_ = optimizer_params_.max_linear_vel_;
  desired_linear_vel_ = optimizer_params_.desired_linear_vel_;

  // Register the dynamic-parameter callback so social profiles can retune the pedestrian-facing
  // weights (+ trajectorizer speed) at runtime via /set_parameters. See header for threading note.
  dyn_params_handler_ = node->add_on_set_parameters_callback(
      std::bind(&SocialMPCController::dynamicParametersCallback, this, std::placeholders::_1));

  // people interface
  people_interface_ = std::make_unique<PeopleInterface>(parent);

  // Track sim pause state (Isaac publishes isaac/sim_running latched) so motion/STOP
  // diagnostics can stay silent while the sim is frozen during a social_yielding pause.
  {
    rclcpp::QoS qos(1);
    qos.transient_local();
    sim_running_sub_ = node->create_subscription<std_msgs::msg::Bool>(
      "/isaac/sim_running", qos,
      [this](std_msgs::msg::Bool::SharedPtr m) { sim_running_ = m->data; });
  }

  // Command-chain taps, fed to the optimizer each cycle for its [MOTION-DIAG] `chain:` fields
  // and `blame=`. Relative names so they resolve inside this robot's
  // namespace, exactly like local_plan below -- an absolute name would break namespaced runs.
  // cmd_vel_nav is our OWN output read back off the topic: it should equal cmd_vx, and a
  // mismatch would mean something republishes/intercepts it, which is worth knowing.
  chain_clock_ = std::make_shared<rclcpp::Clock>(RCL_STEADY_TIME);
  cmd_nav_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel_nav", 1,
    [this](geometry_msgs::msg::Twist::SharedPtr m) {
      last_cmd_nav_vx_ = m->linear.x;
      last_cmd_nav_stamp_ = chain_clock_->now();
    });
  cmd_smoothed_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
    "cmd_vel_smoothed", 1,
    [this](geometry_msgs::msg::Twist::SharedPtr m) {
      last_cmd_smoothed_vx_ = m->linear.x;
      last_cmd_smoothed_stamp_ = chain_clock_->now();
    });
  // ABSOLUTE name on purpose. controller_server is launched with the remap
  // ('cmd_vel' -> 'cmd_vel_nav') that breaks the smoother feedback loop, and a node-level
  // remap rewrites EVERY name the node creates -- including this plugin's subscriptions. A
  // relative "cmd_vel" here would therefore be rewritten to cmd_vel_nav, silently making the
  // `out` tap a duplicate of the `nav` tap and never observing the real end of the chain.
  // Building the name from the node namespace dodges the remap and stays correct for any
  // robot namespace.
  const std::string cmd_out_topic = std::string(node->get_namespace()) + "/cmd_vel";
  cmd_out_sub_ = node->create_subscription<geometry_msgs::msg::Twist>(
    cmd_out_topic, 1,
    [this](geometry_msgs::msg::Twist::SharedPtr m) {
      last_cmd_out_vx_ = m->linear.x;
      last_cmd_out_stamp_ = chain_clock_->now();
    });
  RCLCPP_DEBUG(logger_, "[MOTION-DIAG] chain taps: nav=%s smooth=%s out=%s",
               "cmd_vel_nav (relative)", "cmd_vel_smoothed (relative)", cmd_out_topic.c_str());

  // path handler
  path_handler_ = std::make_unique<mpc::PathHandler>(transform_tolerance_, tf_, costmap_ros_);

  // obstacle distance transform
  obsdist_interface_ =
      std::make_unique<ObstacleDistInterface>(node, costmap_ros_->getGlobalFrameID(), tf_, transform_tolerance_);

  local_path_pub_ = node->create_publisher<nav_msgs::msg::Path>("local_plan", 1);
  people_traj_pub_ = node->create_publisher<visualization_msgs::msg::MarkerArray>("people_projected_trajectory", 1);
}

rcl_interfaces::msg::SetParametersResult
SocialMPCController::dynamicParametersCallback(std::vector<rclcpp::Parameter> parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;

  const std::string weights = plugin_name_ + ".optimizer.weights.";
  const std::string traj_vel = plugin_name_ + ".trajectorizer.desired_linear_vel";
  const std::string max_vel = plugin_name_ + ".optimizer.max_linear_velocity";
  bool optimizer_reinit_needed = false;

  // NOTE: this is a pre-set callback (Humble has no add_post_set_parameters_callback), so the
  // NEW values live in `parameters`, not yet on the node. Read them from `parameters` directly.
  for (const auto& p : parameters)
  {
    const std::string& n = p.get_name();
    if (n == traj_vel && p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    {
      // THE speed knob, and now it reaches both halves of the controller. It feeds the
      // trajectorizer's pure-pursuit reference speed AND the optimizer's VelocityCost target,
      // which used to chase max_linear_vel_ instead -- so before this, pushing a profile's cruise
      // speed only moved the reference path while the cost still pulled toward the ceiling.
      if (trajectorizer_)
        trajectorizer_->setDesiredLinearVel(p.as_double());
      desired_linear_vel_ = p.as_double();
      optimizer_params_.desired_linear_vel_ = p.as_double();
      // A profile push redefines what "unlimited" means, so move the setSpeedLimit baseline with
      // it; otherwise clearing a speed limit later would restore the PREVIOUS profile's speed.
      base_desired_linear_vel_ = p.as_double();
      optimizer_reinit_needed = true;  // optimizer reads its members only at initialize()
      continue;
    }
    if (n == max_vel && p.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
    {
      // Hot-reloadable too, so a profile can bundle the ceiling alongside the cruise speed.
      // This key is NOT under the weights. prefix, so without this branch a runtime push would
      // be accepted by rclcpp and then silently ignored — the optimizer only reads its members
      // at initialize(). Reuse the optimizer_reinit_needed flag to trigger that re-init below.
      optimizer_params_.max_linear_vel_ = p.as_double();
      base_max_linear_vel_ = p.as_double();  // see base_desired_linear_vel_ note above
      optimizer_reinit_needed = true;
      continue;
    }
    if (n.rfind(weights, 0) != 0)
      continue;  // not one of our weight params
    // Defensive: a param callback that throws would break EVERY param-set on controller_server.
    // All weights we hot-reload are doubles; skip anything else under this prefix rather than
    // risk as_double() throwing.
    if (p.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE)
      continue;
    const std::string key = n.substr(weights.size());
    const double v = p.as_double();
    // Pedestrian-facing weights ONLY. obstacle_weight/inflation are intentionally excluded so a
    // social profile never alters static-obstacle avoidance (they are shared across all obstacles).
    if (key == "social_weight")            optimizer_params_.socialwork_w_ = v;
    else if (key == "proxemics_weight")    optimizer_params_.proxemics_w_ = v;
    else if (key == "proxemics_d0")        optimizer_params_.proxemics_d0_ = v;
    else if (key == "proxemics_alpha")     optimizer_params_.proxemics_alpha_ = v;
    else if (key == "social_clear_distance")  optimizer_params_.social_clear_distance_ = v;
    else if (key == "social_safety_distance") optimizer_params_.social_safety_distance_ = v;
    else if (key == "social_mid_gain")     optimizer_params_.social_mid_gain_ = v;
    else if (key == "social_near_gain")    optimizer_params_.social_near_gain_ = v;
    else continue;  // some other weight we don't hot-reload
    optimizer_reinit_needed = true;
  }

  // Re-init the optimizer so it picks up the new values: it reads optimizer_params_ ONLY in
  // initialize(). Covers both the pedestrian weights and optimizer.max_linear_velocity.
  if (optimizer_reinit_needed && optimizer_)
  {
    optimizer_->initialize(optimizer_params_);
    RCLCPP_INFO(logger_, "[social-profile] re-initialized optimizer with updated params at runtime");
  }
  return result;
}

void SocialMPCController::cleanup()
{
  RCLCPP_INFO(logger_,
              "Cleaning up controller: %s of type"
              "nav2_social_mpc_controller::SocialMPCController",
              plugin_name_.c_str());
  local_path_pub_.reset();
  people_traj_pub_.reset();
}

void SocialMPCController::activate()
{
  RCLCPP_INFO(logger_,
              "Activating controller: %s of type "
              "nav2_social_mpc_controller::SocialMPCController",
              plugin_name_.c_str());
  trajectorizer_->activate();
  local_path_pub_->on_activate();
  people_traj_pub_->on_activate();
}

void SocialMPCController::deactivate()
{
  RCLCPP_INFO(logger_,
              "Deactivating controller: %s of type "
              "nav2_social_mpc_controller::SocialMPCController",
              plugin_name_.c_str());
  trajectorizer_->deactivate();
  local_path_pub_->on_deactivate();
  people_traj_pub_->on_deactivate();
}

void SocialMPCController::publish_people_traj(const AgentsTrajectories& people, const std_msgs::msg::Header& header)
{
  // Create one marker for each person
  size_t npeople = people[0].size();
  visualization_msgs::msg::MarkerArray ma;
  for (size_t idx = 0; idx < npeople; idx++)
  {
    if (people[0][idx][3] != -1.0)
    {
      visualization_msgs::msg::Marker m;
      m.header = header;
      m.type = m.LINE_STRIP;
      m.id = idx;
      m.action = m.ADD;
      m.scale.x = 0.05;
      m.color.a = 1.0;
      m.color.r = 1.0;
      m.color.g = 0.0;
      m.color.b = 1.0;
      ma.markers.push_back(m);
    }
  }

  for (unsigned int stepi = 0; stepi < people.size(); stepi++)
  {
    int mi = 0;
    for (unsigned int personi = 0; personi < people[stepi].size(); personi++)
    {
      if (people[stepi][personi][3] != -1.0)
      {
        geometry_msgs::msg::Point point;
        point.x = people[stepi][personi][0];
        point.y = people[stepi][personi][1];
        point.z = 0.1;
        ma.markers[mi].points.push_back(point);
        mi++;
      }
    }
  }
  people_traj_pub_->publish(ma);
}

geometry_msgs::msg::TwistStamped SocialMPCController::computeVelocityCommands(
    const geometry_msgs::msg::PoseStamped& robot_pose, const geometry_msgs::msg::Twist& speed,
    nav2_core::GoalChecker* goal_checker)
{
  nav_msgs::msg::Path transformed_plan =
      path_handler_->transformGlobalPlan(robot_pose, 4.0);  // TODO: make this a parameter

  if (goal_checker != nullptr && !transformed_plan.poses.empty())
  {
    auto & goal_pose = transformed_plan.poses.back().pose;
    if (goal_checker->isGoalReached(robot_pose.pose, goal_pose, speed))
    {
      // Goal reached -> return zero velocity. If this fires mid-route, the
      // windowed goal (truncated plan back()) was wrongly judged "reached".
      double dgoal = std::hypot(goal_pose.position.x - robot_pose.pose.position.x,
                                goal_pose.position.y - robot_pose.pose.position.y);
      RCLCPP_INFO(logger_,
        "[MOTION] state=GOAL | dist_to_windowed_goal=%.3f "
        "odom_vx=%.3f | plan=%zu",
        dgoal, speed.linear.x, transformed_plan.poses.size());
      geometry_msgs::msg::TwistStamped cmd_vel;
      cmd_vel.header = robot_pose.header;
      return cmd_vel;
    }
  }

  auto goal = path_handler_->getTransformedGoal(2.5, transformed_plan, robot_pose);

  // Trajectorize the path
  nav_msgs::msg::Path traj_path = transformed_plan;
  std::vector<geometry_msgs::msg::TwistStamped> cmds;
  // trajectorizer_->trajectorize(traj_path, robot_pose, cmds);

  if (!trajectorizer_->trajectorize(traj_path, robot_pose, cmds))
  {
    geometry_msgs::msg::TwistStamped cmd_vel;
    cmd_vel.header = robot_pose.header;
    cmd_vel.twist.linear.x = 0.0;
    cmd_vel.twist.linear.y = 0.0;
    cmd_vel.twist.angular.z = 0.0;
    RCLCPP_DEBUG(logger_, "Approaching goal without a valid trajectory, stopping");
    return cmd_vel;
  }
  std::vector<geometry_msgs::msg::TwistStamped> init_cmds = cmds;
  // float goal_distance = euclidean_distance(goal.point, robot_pose.pose.position);

  // Be careful, path and people must be in the same frame
  people_msgs::msg::People people_unf = people_interface_->getPeople();
  people_msgs::msg::People people;

  // only use people in the FOV of the robot, in this case (-90°,90° supposed )
  for (auto p : people_unf.people)
  {
    uint mx, my;
    if (!costmap_->worldToMap(p.position.x, p.position.y, mx, my))
    {
      RCLCPP_DEBUG(logger_, "Person %s is not in the costmap", p.name.c_str());
      continue;
    }
    float angle_to_person = atan2(p.position.y - robot_pose.pose.position.y, p.position.x - robot_pose.pose.position.x);
    float robot_yaw = tf2::getYaw(robot_pose.pose.orientation);
    float relative_angle = angles::shortest_angular_distance(robot_yaw, angle_to_person);
    if (fabs(relative_angle) < fov_angle_)
    {
      people.people.push_back(p);
    }
    // Filter people based on the FOV of the robot
  }
  people.header.frame_id = people_unf.header.frame_id;

  if (people.header.frame_id != transformed_plan.header.frame_id)
  {
    // transform people to the global frame.
    // BUG FIX: this loop used `auto p` (a COPY), so `p.position = out_point.point`
    // wrote the transformed coord into the copy and threw it away — people.people
    // kept its ORIGINAL (wrong-frame) coordinates. With /people in `map` and the
    // plan in `<robot>/odom`, a physically-distant pedestrian then landed inside
    // the robot's <2m social_clear_distance in numeric terms, so SocialWorkCost
    // exploded (residual = 700 * |force|^2) and pinned vx→0. Use a reference so
    // the transform is actually written back. Also update the header frame_id.
    for (auto& p : people.people)
    {
      geometry_msgs::msg::PointStamped out_point;
      geometry_msgs::msg::PointStamped in_point;
      in_point.point = p.position;
      in_point.header = people.header;
      if (!transformPoint(transformed_plan.header.frame_id, in_point, out_point))
      {
        throw nav2_core::PlannerException("Unable to transform people point into plan's frame");
      }
      p.position = out_point.point;
    }
    people.header.frame_id = transformed_plan.header.frame_id;
  }

  // Get the distance transform
  obstacle_distance_msgs::msg::ObstacleDistance transformed_od = obsdist_interface_->getDistanceTransform();
  if (!people.people.empty() &&
      (transformed_od.distances.empty() || transformed_od.indexes.empty() || transformed_od.info.width <= 0 ||
       transformed_od.info.height <= 0 || transformed_od.info.resolution <= 0.0))
  {
    RCLCPP_WARN(logger_, "ObstacleDistance is not ready yet, ignoring /people for this control cycle");
    people.people.clear();
  }

  float ts = trajectorizer_->getTimeStep();
  AgentsTrajectories projected_people;

  optimizer_->set_diagnostics_enabled(sim_running_);  // silence [STOP-DIAG] while sim paused
  // Hand the downstream taps to the optimizer so its [MOTION-DIAG] line can name the stage
  // that loses vx. Pushed here, immediately before optimize(), so the values are as fresh as
  // the measured speed they are compared against.
  {
    // NaN age = never received. A large age with a plausible value = STALE, i.e. the tap saw
    // one message long ago and nothing since; that is not the same finding as a live value.
    const auto now_steady = chain_clock_ ? chain_clock_->now() : rclcpp::Time(0, 0, RCL_STEADY_TIME);
    auto age = [&now_steady](const rclcpp::Time& t) {
      return (t.nanoseconds() == 0) ? std::numeric_limits<double>::quiet_NaN()
                                    : (now_steady - t).seconds();
    };
    optimizer_->set_cmd_chain(last_cmd_nav_vx_, last_cmd_smoothed_vx_, last_cmd_out_vx_,
                              age(last_cmd_nav_stamp_), age(last_cmd_smoothed_stamp_),
                              age(last_cmd_out_stamp_));
  }
  bool optimized = optimizer_->optimize(traj_path, projected_people, costmap_, transformed_od, cmds, people, speed, ts);
  if (!optimized)
  {
    RCLCPP_WARN(logger_, "Optimization failed, using initial commands");
    cmds = init_cmds;
  }
  publish_people_traj(projected_people, transformed_plan.header);
  local_path_pub_->publish(traj_path);

  // populate and return twist message
  geometry_msgs::msg::TwistStamped cmd_vel;
  cmd_vel.header = cmds[0].header;
  cmd_vel.twist.linear.x = cmds[0].twist.linear.x;
  cmd_vel.twist.linear.y = 0;
  cmd_vel.twist.angular.z = cmds[0].twist.angular.z;

  // NOTE: the hand-written large-heading in-place-turn override was removed here.
  // Sharp-turn in-place rotation is now owned entirely by the nav2 RotationShimController
  // wrapper (tuned via its `angular_dist_threshold`), so cmd_vel is left as the optimizer
  // output above. This avoids two competing in-place mechanisms fighting on cmd_vel.

  RCLCPP_DEBUG(logger_, "cmd_vel: %f, %f", cmd_vel.twist.linear.x, cmd_vel.twist.angular.z);
  const double out_vx = cmd_vel.twist.linear.x;
  const double out_wz = cmd_vel.twist.angular.z;
  const bool is_stopped = std::fabs(out_vx) < 0.05;

  // Classify the current motion state for a single, human-readable status line.
  const char * state;
  if (!optimized)          state = "OPT_FAIL";   // optimizer failed -> fell back to reference cmds
  else if (!is_stopped)    state = "MOVING";
  else if (std::fabs(out_wz) > 0.1) state = "TURNING";  // ~0 forward but rotating in place
  else                     state = "STOPPED";

  // One status line every cycle (throttled 1s), whatever the state.
  // Suppressed while the sim is paused (social_yielding freeze): the robot is
  // deliberately halted then, so [MOTION]/[MOTION-DIAG] "STOPPED" spam is just noise.
  // Distance to the TRUE final goal (last pose of the un-truncated global plan).
  // transformGlobalPlan() prunes only the PASSED front of global_plan_, never the
  // tail, so global_plan_.back() stays the real destination. transformed_plan.back()
  // is only the local-window end (<= 4m ahead), so it can NOT be used to see "how far
  // from the goal" — this is why the robot could stop 1.68m short and the old
  // plan=%zu line gave no clue. dist_goal is the honest remaining distance.
  double dist_goal = -1.0;
  {
    const auto gplan = path_handler_->getPlan();
    if (!gplan.poses.empty()) {
      dist_goal = std::hypot(gplan.poses.back().pose.position.x - robot_pose.pose.position.x,
                             gplan.poses.back().pose.position.y - robot_pose.pose.position.y);
    }
  }

  static rclcpp::Clock motion_clock(RCL_STEADY_TIME);
  if (sim_running_) {
    RCLCPP_INFO_THROTTLE(logger_, motion_clock, 1000,
      "[MOTION] state=%s vx=%.3f wz=%.3f | ref_vx=%.3f | dist_goal=%.2fm | people=%zu | plan=%zu",
      state, out_vx, out_wz, init_cmds[0].twist.linear.x,
      dist_goal, people.people.size(), transformed_plan.poses.size());
  }

  // On an unexpected stop (or optimizer failure) with a real plan still present,
  // emit one diagnostic line explaining WHY: reference vs output velocities, odom,
  // and the heading error robot->path (a large head_err means we should be turning,
  // not creeping). The per-critic cost breakdown is logged separately by optimizer.cpp.
  if (sim_running_ && (is_stopped || !optimized) && transformed_plan.poses.size() >= 2)
  {
    const double rx = robot_pose.pose.position.x;
    const double ry = robot_pose.pose.position.y;
    const double ryaw = tf2::getYaw(robot_pose.pose.orientation);
    double head_err = 0.0, d_near = -1.0, d_far = -1.0;
    const auto& ps = transformed_plan.poses;
    d_near = std::hypot(ps.front().pose.position.x - rx, ps.front().pose.position.y - ry);
    d_far  = std::hypot(ps.back().pose.position.x - rx,  ps.back().pose.position.y - ry);
    for (const auto& p : ps) {
      double dd = std::hypot(p.pose.position.x - rx, p.pose.position.y - ry);
      if (dd >= 0.5) {
        double bearing = std::atan2(p.pose.position.y - ry, p.pose.position.x - rx);
        head_err = angles::shortest_angular_distance(ryaw, bearing);
        break;
      }
    }
    // dist_goal is carried here too, not only on the [MOTION] INFO line above: that line
    // is INFO and gets filtered out in practice, so the WARN diag was the only thing
    // visible -- and it had no honest "how far from the goal" number. path_end is NOT a
    // substitute: transformed_plan is the local window (<= 4 m ahead), so path_end
    // saturates near 2.5 m and cannot show the robot stalling far from its destination.
    static rclcpp::Clock diag_clock(RCL_STEADY_TIME);
    // DEBUG, not WARN: emitted once a second for the entire run even when the robot is
    // driving correctly, which buries genuine warnings. Re-enable when diagnosing motion
    // with --log-level <controller_logger>:=debug.
    RCLCPP_DEBUG_THROTTLE(logger_, diag_clock, 1000,
      "[MOTION-DIAG] %s: ref_vx=%.3f odom_vx=%.3f odom_wz=%.3f | "
      "head_err_to_path=%.1fdeg | dist_goal=%.2fm | dist path_start=%.2fm path_end=%.2fm | "
      "people=%zu",
      state, init_cmds[0].twist.linear.x, speed.linear.x, speed.angular.z,
      head_err * 180.0 / M_PI, dist_goal, d_near, d_far, people.people.size());
  }
  return cmd_vel;
}

void SocialMPCController::setPlan(const nav_msgs::msg::Path& path)
{
  path_handler_->setPlan(path);
}

void SocialMPCController::setSpeedLimit(const double& speed_limit, const bool& percentage)
{
  // Previously this whole function was inert: it overwrote its own arguments with 0/false, wrote
  // the result into a local named `throwaway_vel`, and logged an unrelated member. Every nav2
  // speed-limiting path (SpeedFilter / speed-restricted zones, and anything else that calls
  // setSpeedLimit) was therefore silently ignored by this controller.
  //
  // Limits are applied against the values from configure(), never against the current effective
  // ones, so repeated calls cannot ratchet the robot down to zero and clearing restores exactly
  // the configured speed.
  double ceiling = base_max_linear_vel_;
  double cruise = base_desired_linear_vel_;

  // nav2 signals "no limit" with 0.0 (NO_SPEED_LIMIT). Compared literally rather than via the
  // constant, which is not exposed by the headers this package pulls in.
  if (speed_limit <= 0.0)
  {
    RCLCPP_INFO(logger_, "[social_mpc] speed limit cleared, restoring max=%.2f cruise=%.2f", ceiling,
                cruise);
  }
  else if (percentage)
  {
    const double frac = std::max(0.0, speed_limit) / 100.0;
    ceiling *= frac;
    cruise *= frac;
    RCLCPP_INFO(logger_, "[social_mpc] speed limit %.1f%% -> max=%.2f cruise=%.2f", speed_limit,
                ceiling, cruise);
  }
  else
  {
    // Absolute m/s. A limit only ever lowers: min() against both, so a limit above the configured
    // speed is a no-op rather than a licence to exceed max_linear_velocity.
    const double lim = std::max(0.0, speed_limit);
    ceiling = std::min(ceiling, lim);
    cruise = std::min(cruise, lim);
    RCLCPP_INFO(logger_, "[social_mpc] speed limit %.2f m/s -> max=%.2f cruise=%.2f", speed_limit,
                ceiling, cruise);
  }

  desired_linear_vel_ = cruise;
  if (trajectorizer_)
    trajectorizer_->setDesiredLinearVel(cruise);
  // Both optimizer copies, or the cost term keeps chasing the unlimited speed.
  optimizer_params_.max_linear_vel_ = ceiling;
  optimizer_params_.desired_linear_vel_ = cruise;
  if (optimizer_)
    optimizer_->initialize(optimizer_params_);  // optimizer reads its members only here
}

bool SocialMPCController::transformPose(const std::string frame, const geometry_msgs::msg::PoseStamped& in_pose,
                                        geometry_msgs::msg::PoseStamped& out_pose) const
{
  if (in_pose.header.frame_id == frame)
  {
    out_pose = in_pose;
    return true;
  }

  try
  {
    tf_->transform(in_pose, out_pose, frame, transform_tolerance_);
    out_pose.header.frame_id = frame;
    return true;
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(logger_, "Exception in transformPose: %s", ex.what());
  }
  return false;
}

bool SocialMPCController::transformPoint(const std::string frame, const geometry_msgs::msg::PointStamped& in_point,
                                         geometry_msgs::msg::PointStamped& out_point) const
{
  try
  {
    tf_->transform(in_point, out_point, frame, transform_tolerance_);
    return true;
  }
  catch (tf2::TransformException& ex)
  {
    RCLCPP_ERROR(logger_, "Exception in transformPoint: %s", ex.what());
  }
  return false;
}

}  // namespace nav2_social_mpc_controller

// Register this controller as a nav2_core plugin
PLUGINLIB_EXPORT_CLASS(nav2_social_mpc_controller::SocialMPCController, nav2_core::Controller)
