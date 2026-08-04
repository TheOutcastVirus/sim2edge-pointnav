# What the scan-to-submap tracker concluded this tick.
#
# The pose is relative to wherever the tracker was last reset, NOT to any map
# or odom frame. This is a short-lived local tracker by design: it has no loop
# closure and no global reference, and publishing it as if it were a map pose
# would invite somebody to use it as one.

std_msgs/Header header

# Relative pose, metres and radians.
float64 x
float64 y
float64 yaw

# Row-major 3x3 over (x, y, yaw).
float64[9] covariance

# Estimated drivetrain scale: achieved forward travel divided by commanded,
# heavily averaged. Drifts away from 1.0 when the wheels slip.
float64 motion_scale

# True when the local geometry made all three planar degrees of freedom
# identifiable, so the scan match was allowed to correct the prior.
#
# This is not a confidence score. A false value means the correction was
# discarded entirely and the pose is prior-only for this tick -- the covariance
# grows to say so, and nothing downstream should treat the pose as lidar-backed.
bool observable

# The correction actually applied after the complementary-filter gain, or zero
# when the scan was rejected.
float64 correction_x
float64 correction_y
float64 correction_yaw

int32 points_used
int32 iterations
int32 submap_points

# "command-gyro" when the prior was integrated from the gyro and the base's
# velocity, "create3-odom" when the base's own odometry increment was used.
# Stamped from what actually happened, so an odometry-assisted run can never be
# reported as lidar/IMU-only evidence.
string motion_prior_source

float64 extract_us
float64 submap_us
float64 icp_us
float64 total_us
