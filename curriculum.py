def get_standing_reward_scales():
    return {
        "survival_time": 8.0,
        "base_height": 5.0,
        "stability": 4.0,
        "energy_efficiency": 2.0,
        "forward_velocity": 2.0
    }


def get_reward_scales(carriculum="standing"):
    # Final reward function
    walk = {
        "survival_time": 8.0,
        "base_height": 5.0,
        "stability": 3.0,
        "energy_efficiency": 2.0,
        "forward_velocity": 12.0,
        "tracking_lin_vel": 8.0,
        "foot_contact": 0.00001,
        "smooth_motion": 0.1,
        "straight_walking": 10.0,
        "large_strides": 10.0,
        "torso_upright": 0.1,
        "crotch_control": 8.0,
    }

    if carriculum == "standing":
        return get_standing_reward_scales()

    return walk
