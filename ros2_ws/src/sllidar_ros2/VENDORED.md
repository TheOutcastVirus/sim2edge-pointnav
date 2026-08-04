# Vendored: `sllidar_ros2`

Slamtec's official RPLIDAR driver, copied into this repository rather than
fetched at build time.

| | |
|---|---|
| Upstream | https://github.com/Slamtec/sllidar_ros2 |
| Branch | `main` |
| Commit | `34300099fadfc772965962dec837bf436706188f` (2024-06-17) |
| License | BSD-2-Clause, see `LICENSE` |
| Local changes | none |

## Why it is vendored and not a `.repos` entry

The board does have outbound HTTPS (ICMP is blocked, which makes `ping
github.com` misleading), so `vcs import` would in fact work. Vendoring is a
choice, not a workaround, for three reasons:

- **One clone is the whole deployment step.** `git clone <this repo> && colcon
  build` on the board, with no `vcs import` and no `rosdep` pass in between.
  That is the point of keeping training and deployment in one repository.
- **The driver version is pinned by the same commit that pins our code.** A
  `.repos` file pins a hash too, but nothing stops a build from running against
  a working copy that drifted; here the driver source is in the diff.
- **The board's network is not something to depend on at build time.** It is on
  a robot, occasionally on a bench, sometimes on the Create 3's USB link with
  Wi-Fi down. A build that needs the internet is a build that fails on the
  day it matters.

The `.git` directory and the two README screenshots were removed; nothing else
was touched. To update, re-clone upstream at the new commit, delete the same
files, and record the new hash above — do not edit the vendored source in place,
or the next update silently discards the change.

## What we actually use

`sllidar_node` publishing `sensor_msgs/LaserScan` on `/scan`, driving an
RPLIDAR A1 over USB serial. The A1 launch file is `launch/sllidar_a1_launch.py`;
`sim2edge_bringup` wraps it with this project's frame and topic conventions
rather than including it directly, so the driver stays a stock upstream package.

Two of its parameters matter more than the rest here:

- `serial_port` — `/dev/ttyUSB0` by default. Nothing is plugged in on the board
  today, so this is untested against the physical unit.
- `scan_mode` — determines beam count and therefore `time_increment`. The
  deskew path reads `time_increment` from the message and falls back to
  spreading `LIDAR_SPIN_HZ` evenly if the driver reports zero, so it degrades
  rather than breaking — but `LIDAR_SPIN_HZ = 5.5` in `sim2edge/spec.py` is a
  *nominal* figure that has never been checked against this hardware. Confirm
  it before quoting any sim-to-real number.
