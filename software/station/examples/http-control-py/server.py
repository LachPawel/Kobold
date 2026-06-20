"""Thin HTTP control server for a NormaCore / ElRobot ST3215 arm.

Holds ONE persistent station client connection and exposes simple HTTP
endpoints so anything (Gemini tool calls, curl, a web page) can drive the arm
with plain JSON — no async-protobuf plumbing on the caller side.

Reads the `st3215/inference` queue (the same one the web UI motor table uses):
each frame carries every bus -> motor with raw present position + calibrated
(range_min, range_max). The bus serial and motor IDs are auto-discovered from
the first frame, so usually you just run `python server.py`.

Run:
    pip install fastapi uvicorn          # or: uv pip install fastapi uvicorn
    python server.py                     # --server localhost, bus auto, motors auto
    # remote daemon:        python server.py --server ab-rpi5.server
    # pin a specific bus:   python server.py --bus-serial 5AB9068903
    # optional camera:      python server.py --video-queue <usbvideo queue id>

The `station` daemon must be running (e.g. `station --tcp`) and the arm
calibrated (so range_min/range_max are real).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path

# --- repo bootstrap so `station_py` and generated protobufs import ----------
import sys
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]  # http-control-py -> examples -> station -> software -> repo
sys.path.insert(0, str(_REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(_REPO))

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from pydantic import BaseModel  # noqa: E402
import uvicorn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("control")

MOTOR_QUEUE = "st3215/inference"
RAM_TORQUE_ENABLE = 0x28
RAM_GOAL_POSITION = 0x2A
RAM_PRESENT_POSITION = 0x38
MAX_STEP = 4095
SIGN_BIT = 0x8000
POSES_PATH = _HERE.parent / "poses.json"


def _u16(b: bytes, addr: int) -> int:
    return struct.unpack_from("<H", b, addr)[0] if len(b) >= addr + 2 else 0


def _present(state: bytes) -> int:
    """Raw present position, sign-bit stripped to 0..4095 (from state.py)."""
    raw = _u16(state, RAM_PRESENT_POSITION)
    if raw & SIGN_BIT:
        return (MAX_STEP + 1 - (raw & MAX_STEP)) & MAX_STEP
    return raw & MAX_STEP


# --- mutable app state ------------------------------------------------------
class State:
    client = None
    want_bus: str = "auto"          # "auto" or a specific serial
    bus_serial: str = ""            # resolved
    motor_ids: list[int] = []       # discovered from frame, in frame order
    latest = None                   # st3215.InferenceStateReader
    latest_stamp: float = 0.0
    video_queue: str = ""
    latest_jpeg: bytes | None = None


S = State()


# --- frame access -----------------------------------------------------------
def _bus():
    if S.latest is None:
        raise HTTPException(503, "no st3215/inference frame yet — is the station running?")
    for b in S.latest.get_buses() or []:
        info = b.get_bus()
        if info and (S.bus_serial == "" or info.get_serial_number() == S.bus_serial):
            return b
    raise HTTPException(503, f"bus '{S.bus_serial}' not in latest frame")


def _motors() -> dict[int, "st3215.InferenceState_MotorStateReader"]:
    return {m.get_id(): m for m in (_bus().get_motors() or [])}


# --- command builders -------------------------------------------------------
def _sync_write(address: int, motors: list[tuple[int, bytes]]) -> commands.DriverCommand:
    cmd = st3215.Command(
        target_bus_serial=S.bus_serial,
        sync_write=st3215.ST3215SyncWriteCommand(
            address=address,
            motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=v) for m, v in motors],
        ),
    )
    return commands.DriverCommand(
        type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode(),
    )


async def _send_ticks(goals: dict[int, int]):
    """Send absolute goal positions (raw ticks), clamped to each calibrated arc."""
    motors = _motors()
    writes = []
    applied = {}
    for mid, tick in goals.items():
        mid = int(mid)
        if mid not in motors:
            raise HTTPException(400, f"motor {mid} not present (have {sorted(motors)})")
        rmin, rmax = int(motors[mid].get_range_min()), int(motors[mid].get_range_max())
        lo, hi = min(rmin, rmax), max(rmin, rmax)
        t = max(lo, min(hi, int(tick)))
        writes.append((mid, t.to_bytes(2, "little")))
        applied[mid] = {"tick": t, "clamped_to": [lo, hi]}
    await send_commands(S.client, [_sync_write(RAM_GOAL_POSITION, writes)])
    return applied


def _norm_to_tick(mid: int, norm: float) -> int:
    m = _motors()[mid]
    rmin, rmax = int(m.get_range_min()), int(m.get_range_max())
    return int(round(rmin + max(0.0, min(1.0, norm)) * (rmax - rmin)))


# --- background follower ----------------------------------------------------
async def _follow_motors():
    q: asyncio.Queue = asyncio.Queue()
    err_q = S.client.follow(MOTOR_QUEUE, q)
    last = b""
    while True:
        if not err_q.empty():
            logger.error("motor stream error: %s", err_q.get_nowait())
        entry = await q.get()
        if entry is None:
            logger.error("motor stream closed")
            return
        eid = bytes(entry.ID.ID)
        if eid == last:
            continue
        last = eid
        S.latest = st3215.InferenceStateReader(memoryview(bytes(entry.Data)))
        S.latest_stamp = time.monotonic()
        if not S.bus_serial:  # one-time resolve on first frame
            for b in S.latest.get_buses() or []:
                info = b.get_bus()
                if info and (S.want_bus in ("auto", info.get_serial_number())):
                    S.bus_serial = info.get_serial_number()
                    S.motor_ids = [m.get_id() for m in (b.get_motors() or [])]
                    logger.info("resolved bus=%s motors=%s", S.bus_serial, S.motor_ids)
                    break


async def _follow_video():
    """Best-effort camera: decode usbvideo FramesPack JPEG frames."""
    from target.gen_python.protobuf.drivers.usbvideo import frame as usbframe  # type: ignore
    q: asyncio.Queue = asyncio.Queue()
    S.client.follow(S.video_queue, q)
    while True:
        entry = await q.get()
        if entry is None:
            return
        try:
            pack = usbframe.FramesPackReader(memoryview(bytes(entry.Data)))
            frames = pack.get_frames_data() or []
            if frames:
                S.latest_jpeg = bytes(frames[0])
        except Exception:
            logger.exception("video decode failed (queue id correct?)")


# --- FastAPI app ------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    S.client = await new_station_client(app.state.server, logger)
    asyncio.create_task(_follow_motors())
    if S.video_queue:
        asyncio.create_task(_follow_video())
    logger.info("connected to %s | video_queue=%s", app.state.server, S.video_queue or "(none)")
    yield


app = FastAPI(title="NormaCore arm control", lifespan=lifespan)


class TorqueReq(BaseModel):
    enable: bool


class MoveReq(BaseModel):
    goals: dict[int, float]      # {motor_id: normalized 0..1}


class MoveRawReq(BaseModel):
    goals: dict[int, int]        # {motor_id: raw tick}


class PoseReq(BaseModel):
    name: str


@app.get("/health")
async def health():
    age = (time.monotonic() - S.latest_stamp) if S.latest_stamp else None
    return {"connected": S.client is not None, "bus": S.bus_serial,
            "motor_ids": S.motor_ids, "frame_age_s": age, "has_video": S.latest_jpeg is not None}


@app.get("/state")
async def state():
    out = []
    for mid, m in _motors().items():
        rmin, rmax = int(m.get_range_min()), int(m.get_range_max())
        present = _present(bytes(m.get_state()))
        span = (rmax - rmin) or 1
        out.append({"motor_id": mid, "present": present, "range_min": rmin, "range_max": rmax,
                    "pct": round((present - rmin) / span, 3)})
    return {"bus": S.bus_serial, "motors": out}


@app.get("/frame.jpg")
async def frame_jpg():
    if S.latest_jpeg is None:
        raise HTTPException(503, "no camera frame. Start server with --video-queue <id>, "
                                 "or feed bridge.py an image with --image.")
    return Response(content=S.latest_jpeg, media_type="image/jpeg")


@app.post("/torque")
async def torque(req: TorqueReq):
    val = b"\x01" if req.enable else b"\x00"
    await send_commands(S.client, [_sync_write(RAM_TORQUE_ENABLE, [(m, val) for m in _motors()])])
    return {"ok": True, "torque": req.enable}


@app.post("/move")
async def move(req: MoveReq):
    return {"ok": True, "applied": await _send_ticks({m: _norm_to_tick(int(m), n) for m, n in req.goals.items()})}


@app.post("/move_raw")
async def move_raw(req: MoveRawReq):
    return {"ok": True, "applied": await _send_ticks(req.goals)}


@app.post("/record_pose")
async def record_pose(req: PoseReq):
    """Snapshot current RAW positions under a name. Disable torque first, hand-pose, then record."""
    snap = {mid: _present(bytes(m.get_state())) for mid, m in _motors().items()}
    poses = json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}
    poses[req.name] = snap
    POSES_PATH.write_text(json.dumps(poses, indent=2))
    return {"ok": True, "name": req.name, "ticks": snap}


@app.post("/replay_pose")
async def replay_pose(req: PoseReq):
    poses = json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}
    if req.name not in poses:
        raise HTTPException(404, f"no pose '{req.name}'. have: {list(poses)}")
    goals = {int(k): int(v) for k, v in poses[req.name].items()}
    return {"ok": True, "name": req.name, "applied": await _send_ticks(goals)}


@app.get("/poses")
async def list_poses():
    return json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="localhost", help="station daemon host")
    p.add_argument("--bus-serial", default="auto", help="ST3215 bus serial, or 'auto' (single bus)")
    p.add_argument("--video-queue", default="", help="optional usbvideo queue id for /frame.jpg")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    S.want_bus = args.bus_serial
    S.video_queue = args.video_queue
    app.state.server = args.server
    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
