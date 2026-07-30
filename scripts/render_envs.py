"""Offscreen screenshots of the parallel environments. No GUI, no viewport.

    ~/isaac-venv/bin/python scripts/render_envs.py --num_envs 16 --out /tmp/envs

Why this exists as a separate script rather than a flag on ``collect_bc.py``:
a collection run launches ``AppLauncher(headless=True)`` with no render
pipeline at all, and there is no way to attach to a live Isaac process and pull
a frame out of it. Getting a picture of a run in flight is impossible; getting a
picture of *the same scene* is cheap, because layouts are a pure function of
``--seed`` and ``--num_layouts``. Pass the collection's seed and the rooms are
identical to the ones it is driving through.

Rendering needs ``enable_cameras=True``, which is what puts the RTX pipeline in
a headless app. Cameras come from ``omni.replicator.core`` render products
rather than a ``ViewportCameraController``: the latter is a UI object and is
``None`` without a GUI, which is exactly the case this script is for.

**This contends for the GPU with a running collector.** It is a few seconds of
render against a job that owns the device for hours, and the RUNBOOK's "do not
run multiple collectors on one GPU" is about sustained throughput, not a
one-shot capture -- but keep ``--num_envs`` small and expect the collector's
``ms/step`` to blip while this runs.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_envs", type=int, default=16,
                    help="Environments to spawn. Kept small on purpose: this is "
                         "a picture, and every env is GPU memory taken from a "
                         "collector that may be running.")
    ap.add_argument("--num_layouts", type=int, default=16,
                    help="Layout pool size. Match the collection run to see its "
                         "rooms; the pool is a pure function of this and --seed.")
    ap.add_argument("--seed", type=int, default=24680)
    ap.add_argument("--out", type=Path, default=Path("renders"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--robot-usd", type=str, default=None,
                    help="Override the robot asset. NavEnvCfg bakes the default "
                         "into UsdFileCfg at class-definition time, so setting "
                         "cfg.robot_usd alone would not take effect.")
    ap.add_argument("--settle", type=int, default=64,
                    help="Render frames to discard before capturing. Materials "
                         "and the BVH stream in asynchronously, so an early "
                         "frame is grey or half-lit rather than wrong-looking.")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    # Same ordering constraint as the collector: the USD bindings are not on
    # sys.path until AppLauncher has run, so nothing under isaaclab.sim -- which
    # includes nav_task -- may be imported above this line.
    from isaaclab.app import AppLauncher
    simulation_app = AppLauncher(headless=True, enable_cameras=True).app

    import numpy as np
    import omni.replicator.core as rep

    from isaaclab_env.nav_task import NavEnvCfg, Sim2EdgeNavEnv

    cfg = NavEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.num_layouts = args.num_layouts
    cfg.layout_seed = args.seed
    cfg.seed = args.seed
    # The collector sets this; it is what puts the robot at a sampled free pose
    # and gives each env a distinct layout rather than a single shared room.
    cfg.collection_mode = True

    if args.robot_usd:
        # Reach into the spawn config, not cfg.robot_usd: that field was read
        # when the class body executed and its value is already inside the
        # UsdFileCfg. Assigning it now changes nothing and looks like it worked.
        cfg.robot_cfg.spawn.usd_path = args.robot_usd
        print(f"[render] robot asset override: {args.robot_usd}", flush=True)

    print(f"[render] building {args.num_envs} envs ...", flush=True)
    env = Sim2EdgeNavEnv(cfg)
    print(f"[render] env ready on {env.device}", flush=True)

    # Reset places robots and assigns layouts. Without it every robot sits at
    # the articulation's init_state and every arena is the same room.
    env.reset()
    print("[render] reset done", flush=True)

    origins = env.scene.env_origins.cpu().numpy()
    args.out.mkdir(parents=True, exist_ok=True)

    # Where the robots actually are, in world coordinates. A screenshot with no
    # robot in it has two very different causes -- not spawned, or spawned
    # outside the frame -- and they are indistinguishable from the image alone.
    robot_w = env.robot.data.root_pos_w.cpu().numpy()
    print(f"[render] robot 0 world {np.round(robot_w[0], 2).tolist()}  "
          f"env 0 origin {np.round(origins[0], 2).tolist()}", flush=True)
    print(f"[render] robot z range [{robot_w[:, 2].min():.3f}, "
          f"{robot_w[:, 2].max():.3f}]  (floor top is z=0)", flush=True)

    shots = _viewpoints(env, origins, robot_w)
    products = []
    for name, eye, lookat in shots:
        cam = rep.create.camera(position=tuple(float(v) for v in eye),
                                look_at=tuple(float(v) for v in lookat))
        rp = rep.create.render_product(cam, (args.width, args.height))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rp])
        products.append((name, annot))
        print(f"[render] camera '{name}' eye {np.round(eye, 1).tolist()} "
              f"-> lookat {np.round(lookat, 1).tolist()}", flush=True)

    # One update() is not enough. The renderer resolves materials, lights and
    # the ray-tracing acceleration structure over several frames, and capturing
    # too early yields a dark or untextured image that reads as a scene bug.
    print(f"[render] settling {args.settle} frames ...", flush=True)
    for _ in range(args.settle):
        simulation_app.update()

    written = []
    for name, annot in products:
        rgb = annot.get_data()
        if rgb is None or getattr(rgb, "size", 0) == 0:
            print(f"[render] WARNING: '{name}' produced no data", flush=True)
            continue
        path = args.out / f"{name}.png"
        _write_png(np.asarray(rgb)[..., :3], path)
        written.append(path)
        print(f"[render] wrote {path}", flush=True)

    print(f"[render] done, {len(written)} image(s)", flush=True)
    env.close()
    simulation_app.close()


def _viewpoints(env, origins, robot_w):
    """(name, eye, lookat) triples in world coordinates.

    Arenas span ``(0,0) -> size_xy`` from their own env origin, so every target
    is an origin plus half an arena -- aiming at the origin itself points at the
    room's corner, which is the documented reason the viewport looks empty.
    """
    import numpy as np

    rooms = [env._rooms[int(i)] for i in env._layout.cpu().numpy()]
    sizes = np.array([r.size_xy for r in rooms], dtype=float)
    centres = origins[:, :2] + sizes / 2.0

    # Overview: everything, from high and to the south so the boxes cast
    # readable shadows instead of flattening into a top-down blueprint.
    lo, hi = centres.min(axis=0), centres.max(axis=0)
    span = float(max(hi - lo).max() if (hi - lo).max() > 0 else sizes.max())
    mid = (lo + hi) / 2.0
    grid = ("grid_overview",
            np.array([mid[0], mid[1] - 0.85 * span, 0.80 * span]),
            np.array([mid[0], mid[1], 0.0]))

    # Close on env 0, matching frame_env0()'s framing in collect_bc.py so the
    # two tools agree on what "looking at an environment" means.
    w, h = float(sizes[0, 0]), float(sizes[0, 1])
    s = max(w, h)
    o = origins[0]
    close = ("env0_close",
             np.array([o[0] + w / 2, o[1] + h / 2 - 1.2 * s, 0.9 * s]),
             np.array([o[0] + w / 2, o[1] + h / 2, 0.0]))

    # A four-env strip: close enough to see robots and goals, wide enough to
    # show that layouts actually differ per environment, which is the thing a
    # screenshot is usually being taken to confirm.
    k = min(4, len(centres))
    sub = centres[:k]
    slo, shi = sub.min(axis=0), sub.max(axis=0)
    sspan = float(max((shi - slo).max(), sizes[:k].max()))
    smid = (slo + shi) / 2.0
    quad = ("first_four",
            np.array([smid[0], smid[1] - 0.9 * sspan, 0.75 * sspan]),
            np.array([smid[0], smid[1], 0.0]))

    # The robot is ~0.35 m across in a 4-10 m arena, so an arena-framed shot
    # renders it at a few dozen pixels against a dark floor -- present, but not
    # obviously so. This one is close enough that "no robot" means no robot.
    # Steeply overhead, not at eye level. A ground-level camera 1.6 m from the
    # robot lands inside whatever obstacle happens to be behind it -- which is
    # what the first version of this did, and the resulting flat white frame
    # reads as "the robot did not render" rather than "the camera is inside a
    # box". From above, only the arena walls could occlude, and they are lower
    # than this.
    r = robot_w[0]
    robot = ("robot_closeup",
             np.array([r[0], r[1] - 1.0, 2.6]),
             np.array([r[0], r[1], 0.1]))

    # Four robots at a distance that still resolves the chassis, to show them
    # in *different* layouts -- which is the actual request a "parallel envs"
    # screenshot is standing in for.
    rk = robot_w[:min(4, len(robot_w))]
    rlo, rhi = rk[:, :2].min(axis=0), rk[:, :2].max(axis=0)
    rmid = (rlo + rhi) / 2.0
    rspan = float(max((rhi - rlo).max(), 6.0))
    robots = ("robots_four",
              np.array([rmid[0], rmid[1] - 0.7 * rspan, 0.55 * rspan]),
              np.array([rmid[0], rmid[1], 0.0]))

    return [grid, close, quad, robot, robots]


def _write_png(rgb, path: Path) -> None:
    """PNG without adding a dependency.

    The project venv has no Pillow and the Isaac venv's copy is an Omniverse
    implementation detail, not something to rely on. zlib and struct are stdlib
    and a PNG is a short, well-specified container.
    """
    import struct
    import zlib

    import numpy as np

    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = a.shape[:2]
    # Filter byte 0 (None) prefixed to each scanline.
    raw = np.hstack([np.zeros((h, 1), np.uint8), a.reshape(h, w * 3)]).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6))
           + chunk(b"IEND", b""))
    path.write_bytes(png)


if __name__ == "__main__":
    main()
