# One control tick of the navigator, for humans and for the log.
#
# Published alongside the command rather than folded into it, so that the thing
# driving the motors carries nothing but the motors' units.

std_msgs/Header header

# The policy's raw output, in SI. The safety node may publish something else.
float64 v
float64 omega

bool has_goal
bool arrived
int32 arrival_streak

# Where the robot BELIEVES the goal is, in its own body frame. The gap between
# this and the truth is the localization error; on hardware there is nothing to
# compare it against, which is the point.
float64 goal_body_x
float64 goal_body_y
float64 belief_distance_m

# The uncertainty bound arrival is decided on: the scan matcher's covariance
# combined with a systematic term proportional to distance travelled. Arrival
# requires belief_distance + 1.96 * sigma to fit inside the arrival radius for
# several consecutive ticks -- the bound, not the point estimate, so the claim
# is falsifiable by the robot's own covariance.
float64 sigma_m
float64 travelled_m

# Sensor freshness at the moment the tick ran, in seconds.
float64 scan_age_s
float64 imu_age_s
float64 odom_age_s

# Why this tick did not command anything, empty when it did.
string hold_reason

# Per-stage cost in microseconds, against a 50 ms period.
float64 deskew_us
float64 bin_us
float64 localize_us
float64 infer_us
float64 total_us

# "qnn", "cpu", or "none".
string backend
