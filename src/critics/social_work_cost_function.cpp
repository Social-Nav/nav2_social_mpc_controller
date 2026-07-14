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

#include "nav2_social_mpc_controller/critics/social_work_cost_function.hpp"

#include <algorithm>

namespace nav2_social_mpc_controller
{

SocialWorkCost::SocialWorkCost(double weight, const AgentsStates& agents_init,
                               const geometry_msgs::msg::Pose& robot_init, const double counter,
                               unsigned int current_position, double time_step, unsigned int control_horizon,
                               unsigned int block_length, double social_clear_distance,
                               double social_safety_distance, double social_mid_gain,
                               double social_near_gain, double social_retreat_gain,
                               double social_retreat_distance)
  : weight_(weight)
  , robot_init_(robot_init)
  , counter_(counter)
  , current_position_(current_position)
  , time_step_(time_step)
  , control_horizon_(control_horizon)
  , block_length_(block_length)
  , social_clear_distance_(social_clear_distance)
  , social_safety_distance_(social_safety_distance)
  , social_mid_gain_(social_mid_gain)
  , social_near_gain_(social_near_gain)
  , social_retreat_gain_(social_retreat_gain)
  , social_retreat_distance_(social_retreat_distance)
{
  for (unsigned int j = 0; j < agents_init.size(); j++)
  {
    original_agents_.col(j) << agents_init[j][0], agents_init[j][1], agents_init[j][2], agents_init[j][3],
        agents_init[j][4], agents_init[j][5];
  }

  sfm_lambda_ = 2.0;
  sfm_gamma_ = 0.35;
  sfm_nPrime_ = 3.0;
  sfm_n_ = 2.0;
  sfm_relaxationTime_ = 0.5;
  sfm_forceFactorSocial_ = 2.1;
  social_clear_distance_ = std::max(social_clear_distance_, social_safety_distance_ + 1e-3);
  social_safety_distance_ = std::max(0.05, social_safety_distance_);
  social_mid_gain_ = std::max(0.0, social_mid_gain_);
  social_near_gain_ = std::max(0.0, social_near_gain_);
  social_retreat_gain_ = std::max(0.0, social_retreat_gain_);
  social_retreat_distance_ = std::max(0.1, social_retreat_distance_);
}

}  // namespace nav2_social_mpc_controller
