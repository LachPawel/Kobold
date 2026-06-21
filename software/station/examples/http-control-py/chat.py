"""Text chat + live camera preview for a NormaCore / ElRobot arm, powered by
Gemini Robotics-ER 1.6.

Open http://localhost:8000 and:
  - ask about what the camera sees ("what's on the table?", "is there a mug?")
  - tell it to point/show ("point at the red object", "show me the laptop")
  - move joints ("rotate the base left a bit", "open the gripper")

One process holds ONE station connection (camera + arm) and talks to ER 1.6.

Run:
    pip install fastapi uvicorn google-genai pillow python-dotenv
    #            (opencv-python only if you want the webcam fallback)
    # put your key in .env (GEMINI_API_KEY=...), then:
    python chat.py
    # remote daemon:  python chat.py --server ab-rpi5.server
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import math
import os
import re
import struct
import time
from contextlib import asynccontextmanager
from pathlib import Path

import sys
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[4]
sys.path.insert(0, str(_REPO / "software" / "station" / "shared"))
sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv
load_dotenv(_HERE.parent / ".env")  # GEMINI_API_KEY etc.

from station_py import new_station_client, send_commands  # noqa: E402
from target.gen_python.protobuf.drivers.st3215 import st3215  # noqa: E402
from target.gen_python.protobuf.station import commands, drivers  # noqa: E402

from fastapi import FastAPI, HTTPException, Response  # noqa: E402
from fastapi.responses import HTMLResponse, FileResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
import uvicorn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("chat")

MODEL_ID = "gemini-robotics-er-1.6-preview"
LOCATE_MODEL = os.getenv("LOCATE_MODEL", "gemini-2.5-flash")  # fast model used inside the servo loop
SERVO_MODEL = os.getenv("SERVO_MODEL", "gemini-robotics-er-1.6-preview")  # ER 1.6 = the vision brain
CAMERA_MODE = os.getenv("CAMERA_MODE", "static")  # 'static' = fixed camera above the arm (eye-to-hand); 'hand' = on the arm
GRIPPER_QUERY = os.getenv("GRIPPER_QUERY", "the robot arm gripper claw")
SERVO_TOL = 75           # tracked-point vs goal distance (0-1000) that counts as "on target"
SERVO_MAX_ITERS = 34     # keep verifying often
SERVO_GAIN = 0.10        # moderate: bigger moves so it actually extends (not too big)
SERVO_MAX_STEP = 0.07    # hard cap on a single iteration's joint move (normalized)
SERVO_SETTLE = 0.5       # seconds per step — faster cadence, still verifies every move
GRIPPER_OPEN = float(os.getenv("GRIPPER_OPEN", "1.0"))   # claw-open  (flipped — open/close was inverted)
GRIPPER_CLOSE = float(os.getenv("GRIPPER_CLOSE", "0.0"))  # claw-closed
# Gemini Live (browser WebSocket voice agent)
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.1-flash-live-preview")
LIVE_SYSTEM = (
    "You are the brain of a NormaCore robot arm, and you SEE its camera feed live (often two views merged side by side). "
    "YOU drive the arm directly with move_joint_delta(motor_id, delta): nudge a joint, watch the NEXT frame, iterate. "
    "JOINT ROLES — use them: M1 = rotate the whole arm left/right (base yaw); M2 = SHOULDER, the BIG reach that swings "
    "the arm forward/back and up/down; M3 = elbow; M4 = forearm extend; M5 & M6 = wrist angle; M7 = wrist twist; M8 = gripper. "
    "For GROSS positioning, lead with M1 (turn toward the target) and M2 (reach/raise toward it) — they move the arm the "
    "most. Then M3/M4 to extend. Use M5-M7 ONLY to fine-tune the gripper angle. Do NOT get stuck only nudging M3-M6. "
    "Use the FULL range of motion: deltas up to ~0.12 (bigger on M1/M2 for large moves, smaller for fine work) — don't "
    "make only tiny middle-of-range moves. Reach in 3D: left/right via M1; forward/back & up/down via M2 then M3/M4. "
    "ALWAYS start a new reach from the safe HOME pose: call go_home first, then move from there. "
    "Use the TOP-DOWN camera as your MAIN reference for the gripper-vs-target offset (left/right + forward/back); use the "
    "other view for up/down. If you canNOT see the target, sweep M1/M2 to look around until it's in frame, then approach. "
    "Iterate until the gripper reaches the target, then stop. open_gripper/close_gripper for the claw, grab to close on an "
    "object, go_home to reset, arm_state to read joint values. Keep spoken replies short; briefly narrate as you move."
)
FUNC_DECLS = [
    {"name": "move_joint_delta", "description": "Nudge ONE joint by a small RELATIVE amount (delta -0.1..0.1 of its range, + or -). Your main control: nudge, watch the camera, repeat. The gripper is motor 8.",
     "parameters": {"type": "object", "properties": {"motor_id": {"type": "integer"}, "delta": {"type": "number"}}, "required": ["motor_id", "delta"]}},
    {"name": "move_joint", "description": "Set one joint to an ABSOLUTE normalized position (0=range min, 1=range max).",
     "parameters": {"type": "object", "properties": {"motor_id": {"type": "integer"}, "normalized": {"type": "number"}}, "required": ["motor_id", "normalized"]}},
    {"name": "arm_state", "description": "Read current joint positions (normalized 0..1) for all motors.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "look", "description": "Describe what is currently visible in the camera (you also see it live, but this gives a careful description).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "set_torque", "description": "Enable (stiffen) or disable (limp) the arm motors.",
     "parameters": {"type": "object", "properties": {"enable": {"type": "boolean"}}, "required": ["enable"]}},
    {"name": "go_home", "description": "Return the arm to its home/rest position.", "parameters": {"type": "object", "properties": {}}},
    {"name": "open_gripper", "description": "Open the claw/gripper.", "parameters": {"type": "object", "properties": {}}},
    {"name": "close_gripper", "description": "Close the claw/gripper to hold an object.", "parameters": {"type": "object", "properties": {}}},
    {"name": "grab", "description": "Pick up an object: open the claw, aim/reach toward the named object, then close the claw.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}},
]
MOTOR_QUEUE = "st3215/inference"
RAM_TORQUE_ENABLE, RAM_GOAL_POSITION, RAM_PRESENT_POSITION = 0x28, 0x2A, 0x38
MAX_STEP, SIGN_BIT = 4095, 0x8000
POSES_PATH = _HERE.parent / "poses.json"
HOME_TICKS = {1: 2100, 2: 2022, 3: 2049, 4: 2065, 5: 2128, 6: 2035, 7: 1981, 8: 1981}  # raw-tick "home" pose
CALIB_PATH = _HERE.parent / "calib.json"   # taught references: [{point:[x,y], ticks:{motor:tick}}]


def _u16(b: bytes, a: int) -> int:
    return struct.unpack_from("<H", b, a)[0] if len(b) >= a + 2 else 0


def _present(state: bytes) -> int:
    raw = _u16(state, RAM_PRESENT_POSITION)
    return ((MAX_STEP + 1 - (raw & MAX_STEP)) & MAX_STEP) if (raw & SIGN_BIT) else (raw & MAX_STEP)


# --- shared state -----------------------------------------------------------
class S:
    client = None
    server = "localhost"
    want_bus = "auto"
    bus_serial = ""
    base_motor = 1          # joint that pans the view left/right (image x)
    tilt_motor = 0          # joint that tilts the view up/down (image y); 0 = disabled
    latest = None
    video_queue = ""        # slot 0 = overhead / top-down camera
    video_queue2 = ""       # slot 1 = wrist camera (optional)
    jpeg: bytes | None = None
    jpeg2: bytes | None = None
    video_tasks: dict = {}   # slot -> asyncio.Task (so we can rebind cameras live)
    llm_src = "0"            # which camera(s) the LLM reasons over: "0", "1", or "merged"
    gemini = None
    # steerable background servo — lets you correct point_at mid-motion by voice
    servo_target = ""
    servo_active = False
    servo_cancel = False
    servo_bias = (0.0, 0.0)   # pixel offset added to the goal, set by nudge()
    servo_status = "idle"


# --- arm helpers ------------------------------------------------------------
def _bus():
    if S.latest is None:
        raise HTTPException(503, "no st3215/inference frame yet — is the station running?")
    for b in S.latest.get_buses() or []:
        info = b.get_bus()
        if info and (S.bus_serial in ("", info.get_serial_number())):
            return b
    raise HTTPException(503, "bus not found in frame")


def _motors():
    return {m.get_id(): m for m in (_bus().get_motors() or [])}


def _sync(addr, motors):
    cmd = st3215.Command(target_bus_serial=S.bus_serial, sync_write=st3215.ST3215SyncWriteCommand(
        address=addr, motors=[st3215.ST3215SyncWriteCommand_MotorWrite(motor_id=m, value=v) for m, v in motors]))
    return commands.DriverCommand(type=drivers.StationCommandType.STC_ST3215_COMMAND, body=cmd.encode())


async def _move_norm(goals: dict[int, float]):
    ms, writes, applied = _motors(), [], {}
    for mid, norm in goals.items():
        mid = int(mid)
        if mid not in ms:
            continue
        rmin, rmax = int(ms[mid].get_range_min()), int(ms[mid].get_range_max())
        lo, hi = min(rmin, rmax), max(rmin, rmax)
        tick = max(lo, min(hi, int(round(rmin + max(0.0, min(1.0, float(norm))) * (rmax - rmin)))))
        writes.append((mid, tick.to_bytes(2, "little")))
        applied[mid] = tick
    if writes:
        await send_commands(S.client, [_sync(RAM_GOAL_POSITION, writes)])
    return applied


async def _torque(enable: bool):
    val = b"\x01" if enable else b"\x00"
    await send_commands(S.client, [_sync(RAM_TORQUE_ENABLE, [(m, val) for m in _motors()])])


async def _replay(name: str):
    poses = json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}
    if name not in poses:
        return {"error": f"no pose '{name}'"}
    ms, writes = _motors(), []
    for k, tick in poses[name].items():
        mid = int(k)
        if mid in ms:
            writes.append((mid, int(tick).to_bytes(2, "little")))
    if writes:
        await send_commands(S.client, [_sync(RAM_GOAL_POSITION, writes)])
    return {"replayed": name}


def _norm_of(mid: int) -> float:
    """Current (actual, measured) joint position as 0..1 within its calibrated arc."""
    m = _motors()[mid]
    rmin, rmax = int(m.get_range_min()), int(m.get_range_max())
    return (_present(bytes(m.get_state())) - rmin) / ((rmax - rmin) or 1)


# --- background followers ---------------------------------------------------
async def _follow_motors():
    q: asyncio.Queue = asyncio.Queue()
    S.client.follow(MOTOR_QUEUE, q)
    last = b""
    while True:
        e = await q.get()
        if e is None:
            logger.error("motor stream closed — reconnect the station?")
            return
        try:
            if bytes(e.ID.ID) == last:
                continue
            last = bytes(e.ID.ID)
            S.latest = st3215.InferenceStateReader(memoryview(bytes(e.Data)))
            if not S.bus_serial:
                for b in S.latest.get_buses() or []:
                    info = b.get_bus()
                    if info and (S.want_bus in ("auto", info.get_serial_number())):
                        S.bus_serial = info.get_serial_number()
                        logger.info("bus=%s motors=%s", S.bus_serial, list(_motors()))
                        break
        except Exception:
            logger.exception("motor frame decode failed (continuing)")


async def _wait_ready(timeout: float = 5.0):
    """Block until the first camera frame + motor frame land (startup race)."""
    deadline = time.monotonic() + timeout
    while (S.jpeg is None or S.latest is None) and time.monotonic() < deadline:
        await asyncio.sleep(0.1)


async def _follow_video(queue: str | None = None, slot: int = 0):
    """Follow a usbvideo queue into camera slot 0 (overhead) or 1 (wrist).

    The queue carries RxEnvelope messages; JPEGs live inside ET_FRAMES envelopes.
    """
    from target.gen_python.protobuf.drivers.usbvideo.usbvideo import (
        RxEnvelopeReader, RxEnvelopeType)
    queue = queue or S.video_queue
    q: asyncio.Queue = asyncio.Queue()
    S.client.follow(queue, q)
    while True:
        e = await q.get()
        if e is None:
            return
        try:
            env = RxEnvelopeReader(memoryview(bytes(e.Data)))
            if env.get_type() != RxEnvelopeType.ET_FRAMES:
                continue
            frames = env.get_frames().get_frames_data() or []
            if frames:
                if slot == 0:
                    S.jpeg = bytes(frames[0])
                else:
                    S.jpeg2 = bytes(frames[0])
        except Exception:
            logger.exception("video decode failed")


def _discover_cameras() -> list[str]:
    """All usbvideo queue ids found on disk (station_data), most-recent first."""
    roots = [Path.home() / "Library/Application Support/@normacore/station-app/station_data",
             Path.cwd() / "station_data"]
    found = []
    for root in roots:
        if not root.is_dir():
            continue
        for cam in sorted({w.parent.name for w in root.rglob("usbvideo/*/wal")}):
            q = f"usbvideo/{cam}"
            if q not in found:
                found.append(q)
    return found


async def _probe_live(queue: str, timeout: float = 3.0) -> bool:
    """Does this queue deliver a frame within `timeout`? (one-shot follow)"""
    from target.gen_python.protobuf.drivers.usbvideo.usbvideo import RxEnvelopeReader, RxEnvelopeType
    q: asyncio.Queue = asyncio.Queue()
    S.client.follow(queue, q)
    try:
        for _ in range(40):
            e = await asyncio.wait_for(q.get(), timeout)
            if e is None:
                return False
            env = RxEnvelopeReader(memoryview(bytes(e.Data)))
            if env.get_type() == RxEnvelopeType.ET_FRAMES and (env.get_frames().get_frames_data() or []):
                return True
    except asyncio.TimeoutError:
        return False
    return False


async def _autobind():
    """A few seconds in, find the cameras that are actually streaming and bind them to
    slots 1 & 2 if the configured ones are dark. USB re-enumeration renames camera queue
    ids on every replug, so this keeps both cameras working without editing .env."""
    await asyncio.sleep(4.0)
    cams = _discover_cameras()
    if not cams:
        return
    results = await asyncio.gather(*[_probe_live(q, 2.5) for q in cams], return_exceptions=True)
    live = [q for q, ok in zip(cams, results) if ok is True]
    if not live:
        return
    if S.jpeg is None:                       # primary dark -> bind first live camera
        logger.warning("primary camera dark — auto-binding %s", live[0])
        await _rebind_camera(0, live[0])
    others = [q for q in live if q != S.video_queue]
    if S.jpeg2 is None and others:           # 2nd dark but another live cam exists -> bind it
        logger.warning("2nd camera dark — auto-binding %s", others[0])
        await _rebind_camera(1, others[0])


async def _rebind_camera(slot: int, queue: str):
    """Point a preview slot at a different camera queue live (no restart)."""
    t = S.video_tasks.get(slot)
    if t:
        t.cancel()
    if slot == 0:
        S.video_queue, S.jpeg = queue, None
    else:
        S.video_queue2, S.jpeg2 = queue, None
    S.video_tasks[slot] = asyncio.create_task(_follow_video(queue, slot))
    return {"slot": slot, "queue": queue}


def _discover_video_queue() -> str:
    """Find a usbvideo queue under the desktop-app (or CLI) station_data dir.

    Layout: .../station_data/<robot-hash>/usbvideo/<camera-id>/wal
    The normfs queue id is `usbvideo/<camera-id>`.
    """
    roots = [Path.home() / "Library/Application Support/@normacore/station-app/station_data",
             Path.cwd() / "station_data"]
    for root in roots:
        if not root.is_dir():
            continue
        cams = sorted({w.parent.name for w in root.rglob("usbvideo/*/wal")})
        if cams:
            if len(cams) > 1:
                logger.info("found %d cameras: %s (set VIDEO_QUEUE in .env to pick another)",
                            len(cams), [f"usbvideo/{c}" for c in cams])
            qid = f"usbvideo/{cams[0]}"
            logger.info("using video queue: %s", qid)
            return qid
    return ""


def _one_frame(slot: int = 0) -> Image.Image | None:
    """A single camera's latest frame as PIL (slot 0 = overhead, 1 = wrist). None if absent.
    Slot 0 falls back to a local webcam if no station frame yet."""
    b = S.jpeg if slot == 0 else S.jpeg2
    if b is not None:
        return Image.open(io.BytesIO(b)).convert("RGB")
    if slot == 0:
        try:
            import cv2  # type: ignore
            cap = getattr(S, "_cap", None) or cv2.VideoCapture(0)
            S._cap = cap
            ok, frame = cap.read()
            if ok:
                ok2, buf = cv2.imencode(".jpg", frame)
                if ok2:
                    S.jpeg = buf.tobytes()
                    return Image.open(io.BytesIO(S.jpeg)).convert("RGB")
        except Exception:
            pass
    return None


def _composite() -> Image.Image:
    """Both camera views merged into ONE labeled image for the LLM (and the preview).
    With one camera, just that frame. Side-by-side, equal height, with TOP/WRIST labels."""
    parts = []
    f0 = _one_frame(0)
    if f0 is not None:
        parts.append(("TOP-DOWN", f0))
    if S.jpeg2 is not None:
        parts.append(("WRIST", _one_frame(1)))
    if not parts:
        raise HTTPException(503, "no camera frame (no station video queue and no webcam).")
    if len(parts) == 1:
        return parts[0][1]
    h = 360
    scaled = [(lbl, im.resize((max(1, int(im.width * h / im.height)), h))) for lbl, im in parts]
    gap, bar = 8, 22
    w = sum(im.width for _, im in scaled) + gap * (len(scaled) - 1)
    canvas = Image.new("RGB", (w, h + bar), (10, 12, 15))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for lbl, im in scaled:
        canvas.paste(im, (x, bar))
        draw.text((x + 4, 5), lbl, fill=(170, 215, 255))
        x += im.width + gap
    return canvas


def _frame_pil() -> Image.Image:
    """What the LLM (look/detect/chat) reasons over — follows the selected mode:
    "0"/"1" = a single camera (fast, no merge); "merged" = both cameras side-by-side."""
    if S.llm_src == "1":
        f = _one_frame(1) or _one_frame(0)
        if f is not None:
            return f
    elif S.llm_src != "merged":          # "0" (default): single primary camera
        f = _one_frame(0)
        if f is not None:
            return f
    return _composite()                  # "merged" (or fallback): both cameras


# --- Gemini ER chat ---------------------------------------------------------
def _parse_json(txt: str) -> str:
    for i, line in enumerate(txt.splitlines()):
        if line.strip() == "```json":
            return "\n".join(txt.splitlines()[i + 1:]).split("```")[0]
    return txt


SCHEMA_HINT = """You are the brain of a NormaCore robot arm with a camera. Reply with ONLY a JSON object:
{
  "reply": "<concise natural-language answer to the user>",
  "points": [{"point":[y,x],"label":"..."}],   // objects to highlight on the live image; [] if none. y,x normalized 0-1000.
  "arm": {"type":"none"}                         // OR one of the actions below
}
Arm actions:
  {"type":"point_at","target":"<object>"}        // rotate the arm toward an object you can see
  {"type":"move_joint","motor_id":<id>,"normalized":<0..1>}  // move one joint (0=range min, 1=range max)
  {"type":"replay_pose","name":"<pose name>"}     // go to a saved pose
  {"type":"none"}
Rules: If the user asks to point at / show / look at an object, set arm.type="point_at" AND include that object in points.
Only use motor_id / pose names that exist (listed below). Keep reply short. Never wrap JSON in code fences."""


def _system_context() -> str:
    try:
        motors = sorted(_motors())
    except Exception:
        motors = []
    poses = list(json.loads(POSES_PATH.read_text())) if POSES_PATH.exists() else []
    return (f"{SCHEMA_HINT}\n\nAvailable motor_ids: {motors} (base/rotation motor = {S.base_motor}, "
            f"gripper is usually the highest id). Saved poses: {poses or 'none'}.")


async def _locate(target: str):
    """Find `target` in the live frame; return its center (y, x) in 0-1000, or (None, None)."""
    img = _frame_pil()
    prompt = (f'Point to the {target}. Return ONLY JSON [{{"point":[y,x],"label":"..."}}], '
              'y,x normalized 0-1000. If it is not visible, return [].')
    cfg = types.GenerateContentConfig(temperature=0.0,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        resp = await asyncio.to_thread(S.gemini.models.generate_content,
                                       model=LOCATE_MODEL, contents=[img, prompt], config=cfg)
        pts = json.loads(_parse_json(resp.text))
        if pts and pts[0].get("point"):
            y, x = pts[0]["point"]
            return float(y), float(x)
    except Exception:
        logger.exception("locate failed")
    return None, None


async def _locate_box(target: str, slot: int = 0, model: str | None = None):
    """Find `target` -> (cx, cy, size_fraction) in 0-1000 / 0..1, or None.

    Uses ONE camera (slot 0 = overhead, 1 = wrist) so coordinates stay unambiguous
    for the servo. `model` overrides the vision model (default SERVO_MODEL = ER 1.6;
    pass LOCATE_MODEL = gemini-2.5-flash for fast, frequent reads like gripper tracking).
    """
    img = _one_frame(slot)
    if img is None:
        return None
    prompt = (f'Find the {target}. Return ONLY JSON: a bounding box [{{"box_2d":[ymin,xmin,ymax,xmax]}}] '
              f'if you can see it clearly, otherwise a point [{{"point":[y,x]}}]. Normalized 0-1000. '
              f'If absent, return [].')
    cfg = types.GenerateContentConfig(temperature=0.0,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        resp = await asyncio.to_thread(S.gemini.models.generate_content,
                                       model=(model or SERVO_MODEL), contents=[img, prompt], config=cfg)
        t = resp.text or ""
        # box (4 numbers) -> center + size; else point (2 numbers) -> center, size unknown
        mb = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", t)
        if mb:
            ymin, xmin, ymax, xmax = (float(mb.group(k)) for k in range(1, 5))
            size = max(0.0, ymax - ymin) * max(0.0, xmax - xmin) / 1.0e6
            return (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, size
        mp = re.search(r"\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]", t)
        if mp:
            y, x = float(mp.group(1)), float(mp.group(2))
            return x, y, None  # point: center known, size unknown
    except Exception:
        logger.exception("locate failed")
    return None


def _servo_joints() -> list[int]:
    """Servos to drive while reaching. Default = every joint except the gripper.
    Override with SERVO_JOINTS=1,2,3,...  (GRIPPER_MOTOR defaults to the highest id)."""
    motors = list(_motors())
    env = os.getenv("SERVO_JOINTS", "").strip()
    if env:
        want = [int(x) for x in env.split(",")]
        return [j for j in want if j in motors]
    return [j for j in motors if j != _gripper_id()]


def _gripper_id() -> int:
    motors = list(_motors())
    return int(os.getenv("GRIPPER_MOTOR", str(max(motors)))) if motors else 0


async def _move_ticks(goals: dict) -> None:
    """Move motors to absolute raw ticks (clamped to each calibrated range)."""
    await _torque(True)
    ms = _motors()
    writes = []
    for mid, tick in goals.items():
        mid = int(mid)
        if mid in ms:
            rmin, rmax = int(ms[mid].get_range_min()), int(ms[mid].get_range_max())
            lo, hi = min(rmin, rmax), max(rmin, rmax)
            writes.append((mid, max(lo, min(hi, int(tick))).to_bytes(2, "little")))
    if writes:
        await send_commands(S.client, [_sync(RAM_GOAL_POSITION, writes)])


def _present_ticks() -> dict:
    return {mid: _present(bytes(m.get_state())) for mid, m in _motors().items()}


def _load_calib() -> list:
    try:
        return json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else []
    except Exception:
        return []


async def _point_calibrated(target: str) -> str:
    """Reliable pointing by taught references: locate target, then move to the
    inverse-distance-weighted blend of the nearest taught poses. No noisy closed loop."""
    cal = _load_calib()
    if len(cal) < 2:
        return None  # not enough references -> caller falls back to the servo
    box = await _locate_box(target, model=LOCATE_MODEL)
    if box is None:
        return f"can't see {target}"
    tx, ty = box[0], box[1]
    near = sorted(cal, key=lambda c: (c["point"][0] - tx) ** 2 + (c["point"][1] - ty) ** 2)[:3]
    weights = [1.0 / (((c["point"][0] - tx) ** 2 + (c["point"][1] - ty) ** 2) + 1.0) for c in near]
    goals = {}
    for mid in _motors():
        pairs = [(c["ticks"].get(str(mid)), w) for c, w in zip(near, weights) if c["ticks"].get(str(mid)) is not None]
        if pairs:
            goals[mid] = int(sum(v * w for v, w in pairs) / sum(w for _, w in pairs))
    await _move_ticks(goals)
    return f"pointed at {target} using {len(near)} taught references"


async def _go_home() -> str:
    """Move every joint to the saved home (entry) pose, in raw ticks (clamped to range)."""
    await _stop_servo()
    await _move_ticks(HOME_TICKS)
    return "moved to home position"


async def _set_gripper(open_it: bool) -> str:
    await _torque(True)
    g = _gripper_id()
    await _move_norm({g: GRIPPER_OPEN if open_it else GRIPPER_CLOSE})
    return f"claw {'opened' if open_it else 'closed'} (M{g})"


async def _servo_loop():
    """Steerable background visual servo (ER 1.6 vision). Runs while S.servo_active,
    reading S.servo_target every iteration so VOICE can correct it mid-motion:
      - retarget: name a different object -> S.servo_target changes -> loop re-aims
      - nudge:    S.servo_bias shifts the goal pixel (e.g. "a bit left")
      - stop:     S.servo_cancel ends it

    static cam (eye-to-hand): drive the GRIPPER onto the TARGET pixel.
    hand cam   (eye-in-hand):  drive the TARGET to frame center.
    Coordinate descent over non-gripper joints: small step, verify next frame,
    step back + reverse if worse, freeze joints with no effect.
    """
    active = _servo_joints()
    if not active:
        S.servo_status = "no servo joints"; S.servo_active = False; return
    await _torque(True)
    cur_target = S.servo_target
    sign = {j: 1 for j in active}; gain = {j: SERVO_GAIN for j in active}
    frozen: set[int] = set(); prev_pos: dict[int, float] = {}
    prev_cost = None; last_j = None; qi = 0; goal = None; grip = None; i = 0
    while S.servo_active and not S.servo_cancel and i < SERVO_MAX_ITERS:
        if S.servo_target != cur_target:                   # voice retargeted mid-motion
            cur_target = S.servo_target
            sign = {j: 1 for j in active}; gain = {j: SERVO_GAIN for j in active}
            frozen = set(); prev_cost = None; last_j = None; goal = None; grip = None; i = 0
            S.servo_status = f"re-aiming at {cur_target}"
        bx, by = S.servo_bias
        if CAMERA_MODE == "static":
            if goal is None or i % 3 == 0:                 # re-check the target often (it may shift)
                tb = await _locate_box(cur_target)
                if tb:
                    goal = tb[:2] if goal is None else (0.5 * tb[0] + 0.5 * goal[0], 0.5 * tb[1] + 0.5 * goal[1])
            tg = await _locate_box(GRIPPER_QUERY, model=LOCATE_MODEL)   # fast 2.5-flash for frequent gripper reads
            if tg is not None:                             # EMA-smooth the noisy gripper reads -> smooth motion
                grip = tg[:2] if grip is None else (0.6 * tg[0] + 0.4 * grip[0], 0.6 * tg[1] + 0.4 * grip[1])
            if goal is None:
                S.servo_status = f"can't see {cur_target}"; await asyncio.sleep(SERVO_SETTLE); continue
            if grip is None:
                S.servo_status = "can't see the gripper"; await asyncio.sleep(SERVO_SETTLE); continue
            ex, ey = (goal[0] + bx) - grip[0], (goal[1] + by) - grip[1]
        else:
            tb = await _locate_box(cur_target)
            if tb is None:
                S.servo_status = f"can't see {cur_target}"; await asyncio.sleep(SERVO_SETTLE); continue
            ex, ey = (500.0 + bx) - tb[0], (500.0 + by) - tb[1]
        err = math.hypot(ex, ey); cost = err / 707.0
        S.servo_status = f"aiming at {cur_target}, off by {int(err)}"
        if err < SERVO_TOL:
            S.servo_status = f"on target ({cur_target})"; break
        if last_j is not None and prev_cost is not None:
            dcost = cost - prev_cost
            if dcost > 0.012:                      # worse -> step back + reverse
                sign[last_j] *= -1; gain[last_j] = max(0.045, gain[last_j] * 0.7)
                await _move_norm({last_j: prev_pos[last_j]})
            elif abs(dcost) < 0.008:               # no effect -> freeze late
                gain[last_j] *= 0.6
                if gain[last_j] < 0.02:
                    frozen.add(last_j)
        prev_cost = cost
        avail = [j for j in active if j not in frozen]
        if not avail:
            S.servo_status = f"converged on {cur_target}"; break
        j = avail[qi % len(avail)]; qi += 1
        cur = _norm_of(j); prev_pos[j] = cur; last_j = j
        scale = max(0.55, min(1.0, cost))
        d = max(-SERVO_MAX_STEP, min(SERVO_MAX_STEP, sign[j] * gain[j] * scale))
        await _move_norm({j: max(0.0, min(1.0, cur + d))})
        await asyncio.sleep(SERVO_SETTLE)
        i += 1
    if S.servo_cancel:
        S.servo_status = "stopped"
    S.servo_active = False


async def _start_servo(target: str) -> str:
    """Start (or, if already running, retarget) the steerable servo. Returns immediately
    so the voice layer keeps listening and can correct it mid-motion."""
    target = (target or "").strip()
    if not target:
        return "no target given"
    S.servo_target = target
    S.servo_bias = (0.0, 0.0)
    if S.servo_active:
        return f"now aiming at {target}"
    S.servo_cancel = False
    S.servo_active = True
    asyncio.create_task(_servo_loop())
    return f"reaching for {target} — say 'stop', name another object, or 'a bit left/right' to correct"


async def _stop_servo() -> str:
    if S.servo_active:
        S.servo_cancel = True
        for _ in range(25):
            if not S.servo_active:
                break
            await asyncio.sleep(0.1)
    return "stopped"


async def _wait_servo_done(timeout: float = 60.0):
    deadline = time.monotonic() + timeout
    while S.servo_active and time.monotonic() < deadline:
        await asyncio.sleep(0.2)


async def _servo_to(target: str) -> str:
    """Blocking point-at: start the loop and wait for it to settle (used by grab)."""
    await _stop_servo()
    await _start_servo(target)
    await _wait_servo_done()
    return S.servo_status


async def _exec_arm(arm: dict, points: list) -> str:
    t = (arm or {}).get("type", "none")
    if t == "none":
        return ""
    await _torque(True)  # the arm can't move while limp; enable before any motion
    if t == "move_joint":
        applied = await _move_norm({int(arm["motor_id"]): float(arm["normalized"])})
        return f"moved joint {arm['motor_id']} -> {applied}"
    if t == "replay_pose":
        return json.dumps(await _replay(arm["name"]))
    if t == "point_at":
        target = (arm.get("target") or "").strip()
        if not target and points:
            target = points[0].get("label", "")
        if not target:
            return "no target to point at"
        return await _servo_to(target)
    return ""


async def _run_tool(name: str, args: dict) -> str:
    """Execute one Live tool call against the arm (shared by the web WS client and live.py)."""
    try:
        if name == "look":
            img = _frame_pil()
            cfg = types.GenerateContentConfig(temperature=0.4, thinking_config=types.ThinkingConfig(thinking_budget=0))
            r = await asyncio.to_thread(S.gemini.models.generate_content, model=MODEL_ID,
                                        contents=[img, "Briefly describe the scene and list the notable objects you see."], config=cfg)
            return (r.text or "").strip()[:500]
        if name == "point_at" or name == "retarget":
            target = (args.get("target") or "").strip()
            done = await _point_calibrated(target)   # reliable taught-reference replay (if calibrated)
            if done is not None:
                return done
            return await _start_servo(target)        # else fall back to the closed-loop servo
        if name == "stop":
            return await _stop_servo()
        if name == "nudge":
            d = (args.get("direction") or "").lower()
            bx, by = S.servo_bias
            step = 130.0
            if "left" in d: bx -= step
            elif "right" in d: bx += step
            elif "up" in d or "forward" in d: by -= step
            elif "down" in d or "back" in d: by += step
            S.servo_bias = (bx, by)
            return f"nudging {d or 'nowhere'} (bias now {int(bx)},{int(by)})"
        if name == "move_joint_delta":
            await _torque(True)
            mid = int(args["motor_id"]); delta = float(args["delta"])
            cur = _norm_of(mid)
            nxt = max(0.0, min(1.0, cur + delta))
            await _move_norm({mid: nxt})
            return f"joint {mid}: {cur:.2f} -> {nxt:.2f}"
        if name == "arm_state":
            return json.dumps({str(mid): round(_norm_of(mid), 2) for mid in _motors()})
        if name == "move_joint":
            await _torque(True)
            return f"moved joint {args['motor_id']} -> {await _move_norm({int(args['motor_id']): float(args['normalized'])})}"
        if name == "replay_pose":
            await _torque(True)
            return json.dumps(await _replay(args["name"]))
        if name == "set_torque":
            await _torque(bool(args["enable"]))
            return "torque " + ("enabled" if args["enable"] else "disabled (limp)")
        if name == "go_home":
            return await _go_home()
        if name == "open_gripper":
            return await _set_gripper(True)
        if name == "close_gripper":
            return await _set_gripper(False)
        if name == "grab":
            t = (args.get("target") or "").strip()
            if not t:
                return "no target to grab"
            await _stop_servo()             # cancel any running point-at first
            await _set_gripper(True)        # open the claw
            servo_log = await _servo_to(t)  # aim + reach toward it (blocking)
            await _set_gripper(False)       # close on it
            return f"grab '{t}': {servo_log} | claw closed"
    except Exception as e:
        logger.exception("tool %s failed", name)
        return f"error: {e}"
    return f"unknown tool {name}"


async def chat_turn(message: str) -> dict:
    await _wait_ready()
    img = _frame_pil()
    prompt = f"{_system_context()}\n\nUser: {message}"
    cfg = types.GenerateContentConfig(temperature=0.4,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        resp = await asyncio.to_thread(
            S.gemini.models.generate_content, model=MODEL_ID, contents=[img, prompt], config=cfg)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
            import re
            m = re.search(r"retry in ([\d.]+)s", msg, re.I) or re.search(r"'retryDelay': '(\d+)s'", msg)
            wait = f" Try again in ~{int(float(m.group(1)))}s." if m else ""
            return {"reply": f"I've hit the Gemini free-tier limit for ER 1.6 (20 requests/day)."
                             f" Enable billing on the API key for real use.{wait}",
                    "points": [], "action": "rate-limited (429)"}
        logger.exception("Gemini call failed")
        return {"reply": f"Model error: {msg[:160]}", "points": [], "action": "error"}
    try:
        data = json.loads(_parse_json(resp.text))
    except Exception:
        return {"reply": resp.text, "points": [], "action": ""}
    try:
        action_log = await _exec_arm(data.get("arm", {}), data.get("points", []))
    except Exception as e:
        action_log = f"arm error: {str(e)[:120]}"
    return {"reply": data.get("reply", ""), "points": data.get("points", []), "action": action_log}


# --- FastAPI ----------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    S.client = await new_station_client(S.server, logger)
    S.gemini = genai.Client()  # reads GEMINI_API_KEY from .env
    asyncio.create_task(_follow_motors())
    if not S.video_queue:
        S.video_queue = _discover_video_queue()
    if S.video_queue:
        S.video_tasks[0] = asyncio.create_task(_follow_video(S.video_queue, 0))   # overhead
    else:
        logger.warning("no station video queue found — will try local webcam for the preview")
    if S.video_queue2:
        S.video_tasks[1] = asyncio.create_task(_follow_video(S.video_queue2, 1))  # wrist
        logger.info("wrist camera: %s", S.video_queue2)
    asyncio.create_task(_autobind())   # self-heal if the configured primary is dark
    yield


app = FastAPI(lifespan=lifespan)


class ChatReq(BaseModel):
    message: str


@app.get("/favicon.ico")
async def favicon():
    return Response(status_code=204)


@app.get("/api/health")
async def health():
    motors = []
    try:
        motors = list(_motors())
    except Exception:
        pass
    return {"motor_frame": S.latest is not None, "bus": S.bus_serial, "motor_ids": motors,
            "has_camera": S.jpeg is not None, "video_queue": S.video_queue}


@app.get("/api/frame.jpg")
async def frame(src: str = "merged"):
    # src: "merged" (both), "0" (slot 0 only), "1" (slot 1 only).
    # Never hard-fail if SOMETHING is available: fall back to the other slot / merged.
    pil = None
    if src in ("0", "1"):
        pil = _one_frame(int(src)) or _one_frame(1 - int(src))
    if pil is None:
        try:
            pil = _composite()
        except HTTPException:
            raise HTTPException(503, "no camera has frames yet — click ↻ then pick a live camera (→1)")
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


class CamReq(BaseModel):
    slot: int = 0
    queue: str


@app.get("/api/cameras")
async def cameras():
    """All discovered camera queues + which slot each is bound to + whether it has a frame."""
    bound = {S.video_queue: 0, S.video_queue2: 1}
    discovered = _discover_cameras()
    # include bound queues even if discovery missed them
    for q in (S.video_queue, S.video_queue2):
        if q and q not in discovered:
            discovered.append(q)
    return {"cameras": [{"queue": q, "slot": bound.get(q),
                         "has_frame": (S.jpeg is not None) if bound.get(q) == 0 else
                                      (S.jpeg2 is not None) if bound.get(q) == 1 else None}
                        for q in discovered],
            "slot0": S.video_queue, "slot1": S.video_queue2}


@app.post("/api/setcam")
async def setcam(req: CamReq):
    return await _rebind_camera(req.slot, req.queue)


@app.post("/api/view/{src}")
async def set_view(src: str):
    """Pick which camera(s) the LLM reasons over: '0', '1', or 'merged' (both)."""
    S.llm_src = src if src in ("0", "1", "merged") else "0"
    return {"llm_src": S.llm_src}


class CalibReq(BaseModel):
    x: float
    y: float


@app.get("/api/calib")
async def calib_list():
    return {"refs": _load_calib()}


@app.post("/api/calib")
async def calib_add(req: CalibReq):
    """Teach a reference: pair the clicked image point [x,y] (0-1000) with the arm's
    CURRENT joint pose (limp the arm and hand-point it first). Records present ticks."""
    cal = _load_calib()
    cal.append({"point": [req.x, req.y], "ticks": {str(k): v for k, v in _present_ticks().items()}})
    CALIB_PATH.write_text(json.dumps(cal, indent=2))
    return {"count": len(cal), "refs": cal}


@app.post("/api/calib/clear")
async def calib_clear():
    CALIB_PATH.write_text("[]")
    return {"count": 0}


@app.post("/api/chat")
async def chat(req: ChatReq):
    try:
        return await chat_turn(req.message)
    except Exception as e:
        logger.exception("chat failed")
        return {"reply": f"Error: {str(e)[:160]}", "points": [], "action": "error"}


class ToolReq(BaseModel):
    name: str
    args: dict = {}


@app.get("/api/live-config")
async def live_config():
    # localhost demo: hands the key to the browser so it can open the Live WS directly.
    # For production, swap to ephemeral tokens (https://ai.google.dev/gemini-api/docs/ephemeral-tokens).
    return {"apiKey": os.getenv("GEMINI_API_KEY", ""), "model": LIVE_MODEL,
            "system": LIVE_SYSTEM, "decls": FUNC_DECLS,
            "aicLicense": os.getenv("AIC_SDK_LICENSE", ""),
            "aicModelUrl": os.getenv("AIC_MODEL_URL", "/api/vf-model")}  # served locally (Quail VF 2.1, v5)


@app.get("/api/vf-model")
async def vf_model():
    """Serve the local Quail Voice Focus .aicmodel (downloaded via the Python SDK) so the
    browser WASM loads a version-compatible model from our own origin (no CORS, no version guessing)."""
    files = sorted((_HERE.parent / "models").glob("*.aicmodel"))
    if not files:
        raise HTTPException(404, "no .aicmodel in ./models — run: "
            "uv run python -c \"import aic_sdk,dotenv;dotenv.load_dotenv('.env');aic_sdk.Model.download('quail-vf-2.1-l-16khz','./models')\"")
    return FileResponse(str(files[0]), media_type="application/octet-stream")


@app.post("/api/tool")
async def api_tool(req: ToolReq):
    return {"result": await _run_tool(req.name, req.args)}


@app.get("/api/state")
async def api_state():
    """Joint positions + calibrated ranges (for an external brain like Claude/Codex)."""
    out = []
    try:
        for mid, m in _motors().items():
            out.append({"motor_id": mid, "present_norm": round(_norm_of(mid), 3),
                        "range_min": int(m.get_range_min()), "range_max": int(m.get_range_max())})
    except Exception as e:
        return {"motors": [], "error": str(e)[:120]}
    return {"bus": S.bus_serial, "gripper_motor": _gripper_id(), "motors": out}


@app.get("/api/detect")
async def detect():
    """ER 1.6 object detection for the live overlay: list of {box, label}."""
    img = _frame_pil()
    prompt = ('Detect the prominent objects in view. Return ONLY a JSON array '
              '[{"box_2d":[ymin,xmin,ymax,xmax],"label":"<name>"}], normalized 0-1000, integers, '
              'at most 12 objects, no masks, no code fencing.')
    cfg = types.GenerateContentConfig(temperature=0.0, thinking_config=types.ThinkingConfig(thinking_budget=0))
    boxes = []
    try:
        r = await asyncio.to_thread(S.gemini.models.generate_content, model=MODEL_ID, contents=[img, prompt], config=cfg)
        txt = _parse_json(r.text or "")
        a, b = txt.find("["), txt.rfind("]")
        if a != -1 and b > a:
            try:
                for it in json.loads(txt[a:b + 1]):
                    bb = it.get("box_2d")
                    if bb and len(bb) == 4:
                        boxes.append({"box": [int(v) for v in bb], "label": it.get("label", "")})
            except Exception:
                for m in re.finditer(r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]", txt):
                    boxes.append({"box": [int(m.group(k)) for k in range(1, 5)], "label": ""})
    except Exception as e:
        logger.exception("detect failed")
        return {"boxes": [], "error": str(e)[:120]}
    return {"boxes": boxes[:12]}


@app.post("/api/torque/{enable}")
async def torque(enable: int):
    await _torque(bool(enable))
    return {"ok": True, "torque": bool(enable)}


@app.post("/api/record/{name}")
async def record(name: str):
    snap = {mid: _present(bytes(m.get_state())) for mid, m in _motors().items()}
    poses = json.loads(POSES_PATH.read_text()) if POSES_PATH.exists() else {}
    poses[name] = snap
    POSES_PATH.write_text(json.dumps(poses, indent=2))
    return {"ok": True, "name": name, "ticks": snap}


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Kobold · robot assistant</title>
<style>
 :root{--bg:#0F0F0F;--panel:#1A1919;--pri:#F9F9F9;--sec:#DEDEDE;--ter:#666;--border:#333;
   --accent:#6993FF;--glow:#006FFF;--green:#00BFA6;--orange:#F4B942;--red:#E28C7C;
   --mono:ui-monospace,SFMono-Regular,Menlo,monospace;color-scheme:dark}
 *{box-sizing:border-box}
 body{margin:0;font:14px/1.5 Inter,-apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);
   color:var(--pri);height:100vh;display:flex;flex-direction:column;overflow:hidden;-webkit-font-smoothing:antialiased}
 .glow{position:fixed;inset:0;overflow:hidden;pointer-events:none;z-index:0}
 .glow::before{content:"";position:absolute;left:28%;top:-260px;width:1100px;height:760px;
   background:radial-gradient(ellipse,var(--glow) 0%,transparent 70%);filter:blur(110px);opacity:.13}
 .topbar{position:relative;z-index:2;display:flex;align-items:center;justify-content:space-between;
   padding:13px 22px;border-bottom:1px solid var(--border)}
 .brand{display:inline-flex;align-items:center;gap:9px;font-weight:600;font-size:15px}
 .brand .mark{width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 12px var(--glow)}
 .brand .sub{color:var(--ter);font-weight:400}
 .badge{display:inline-flex;align-items:center;gap:8px;padding:6px 14px;border-radius:999px;font-size:11px;
   font-weight:600;text-transform:uppercase;letter-spacing:.8px;background:var(--panel);border:1px solid var(--border);color:var(--ter)}
 .badge .dot{width:8px;height:8px;border-radius:50%;background:var(--ter)}
 .badge.live{color:var(--green);border-color:rgba(0,191,166,.4);background:rgba(0,191,166,.06)}
 .badge.live .dot{background:var(--green);box-shadow:0 0 12px var(--green);animation:pulse 1.4s infinite}
 .badge.err{color:var(--red);border-color:rgba(226,140,124,.4)} .badge.err .dot{background:var(--red)}
 main{position:relative;z-index:1;flex:1;display:flex;min-height:0}
 #left{flex:1;padding:18px;display:flex;flex-direction:column;gap:13px;min-width:0}
 .cap{font-size:10px;text-transform:uppercase;letter-spacing:.9px;color:var(--ter)}
 #stage{position:relative;background:#000;border:1px solid var(--border);border-radius:14px;overflow:hidden;
   flex:1;display:flex;align-items:center;justify-content:center}
 #cam{max-width:100%;max-height:100%;display:block}
 #overlay,#detov{position:absolute;inset:0;pointer-events:none}
 .pt{position:absolute;width:14px;height:14px;border-radius:50%;background:var(--accent);border:2px solid #fff;
   transform:translate(-50%,-50%);box-shadow:0 0 22px rgba(105,147,255,.7)}
 .lbl{position:absolute;background:var(--accent);color:#08122e;font-size:12px;font-weight:600;padding:2px 8px;
   border-radius:5px;transform:translate(12px,-10px);white-space:nowrap}
 .dbox{position:absolute;border:2px solid var(--green);border-radius:4px;box-shadow:0 0 12px rgba(0,191,166,.35)}
 .dlbl{position:absolute;top:-17px;left:-2px;background:var(--green);color:#04140d;font-size:11px;font-weight:600;
   padding:1px 6px;border-radius:4px;white-space:nowrap}
 #ctrls{display:flex;gap:8px;flex-wrap:wrap}
 .chip{background:var(--panel);color:var(--sec);border:1px solid var(--border);border-radius:8px;padding:8px 12px;
   cursor:pointer;font:inherit;font-size:12px;transition:border-color .15s,color .15s}
 .chip:hover{border-color:var(--accent);color:var(--pri)}
 .chip.on{border-color:var(--accent);color:var(--pri);background:#1f2535}
 .camrow{display:flex;justify-content:space-between;align-items:center;gap:10px;flex-wrap:wrap}
 .camctl{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
 .camctl .chip{padding:5px 8px;font-size:11px}
 #camsel{max-width:160px;font-size:11px}
 #detbtn.on{background:#0f3d33;border-color:var(--green);color:#d6fff0}
 #right{width:430px;border-left:1px solid var(--border);display:flex;flex-direction:column;background:rgba(255,255,255,.012)}
 #log{flex:1;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:10px}
 .msg{padding:10px 13px;border-radius:12px;max-width:88%;font-size:13px;line-height:1.5}
 .me{align-self:flex-end;background:var(--accent);color:#08122e}
 .bot{align-self:flex-start;background:var(--panel);border:1px solid var(--border);color:var(--sec)}
 .act{align-self:flex-start;font-size:11px;color:var(--ter);font-family:var(--mono);white-space:pre-wrap;word-break:break-word}
 #bar{display:flex;gap:8px;padding:14px;border-top:1px solid var(--border);align-items:center}
 #inp{flex:1;background:#0a0a0d;border:1px solid var(--border);border-radius:10px;color:#fff;padding:11px 13px;font:inherit}
 #inp:focus{outline:none;border-color:var(--accent)}
 .iconbtn{font-size:12px;font-weight:600;min-width:46px;height:42px;padding:0 10px;flex:none;display:inline-flex;align-items:center;justify-content:center;
   background:var(--panel);border:1px solid var(--border);border-radius:10px;cursor:pointer;color:var(--sec);transition:all .15s}
 .iconbtn:hover{border-color:var(--accent);color:var(--pri)}
 #livebtn.on{background:var(--green);border-color:var(--green);color:#04140d;animation:pulse 1.4s infinite}
 #micbtn.on{background:var(--red);border-color:var(--red);color:#fff;animation:pulse 1.1s infinite}
 #spkbtn.on{background:var(--green);border-color:var(--green);color:#04140d}
 #vfbtn.on{background:var(--orange);border-color:var(--orange);color:#1a1505}
 .send{background:var(--accent);border:0;color:#08122e;font-weight:600;border-radius:10px;padding:11px 16px;cursor:pointer;font:inherit}
 .send:hover{filter:brightness(1.08)}
 @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(105,147,255,.5)}50%{box-shadow:0 0 0 9px rgba(105,147,255,0)}}
</style></head><body>
<div class=glow></div>
<header class=topbar>
 <span class=brand><span class=mark></span>Kobold<span class=sub>· robot assistant</span></span>
 <span id=status class=badge><span class=dot></span>Connecting…</span>
</header>
<main>
<div id=left>
 <div class=camrow>
   <span class=cap>Live camera · ER 1.6</span>
   <span class=camctl>
     <select id=camsel class=chip title="Discovered cameras"></select>
     <button class=chip onclick="pickCam(0)" title="Use as primary (slot 1)">→1</button>
     <button class=chip onclick="pickCam(1)" title="Use as 2nd view (slot 2)">→2</button>
     <button class=chip onclick="loadCams()" title="Rescan cameras">↻</button>
     <button class=chip id=vboth onclick="setView('merged')">Both</button>
     <button class=chip id=vtop onclick="setView('0')">1</button>
     <button class=chip id=vwrist onclick="setView('1')">2</button>
     <button class=chip id=teachbtn onclick="teachToggle()" title="Teach a reference: limp the arm, point it at a spot, click that spot">Teach</button>
     <button class=chip onclick="clearRefs()" title="Clear taught references">Clear refs</button>
   </span>
 </div>
 <div id=stage><img id=cam><div id=overlay></div><div id=detov></div></div>
 <div id=ctrls>
   <button class=chip onclick="callTool('go_home')">Home</button>
   <button id=detbtn class=chip onclick="detToggle()">ER overlay</button>
   <button class=chip onclick="callTool('open_gripper')">Open claw</button>
   <button class=chip onclick="callTool('close_gripper')">Close claw</button>
   <button class=chip onclick="t(0)">Torque off</button>
   <button class=chip onclick="t(1)">Torque on</button>
   <button class=chip onclick="rec()">Record pose</button>
 </div>
</div>
<div id=right>
 <div id=log><div class=msg bot>Hi. I see through the arm's camera. Click Live or type — ask what I see, or tell me to point at something.</div></div>
 <div id=bar>
   <button id=livebtn class=iconbtn onclick=liveToggle() title="Native voice — Gemini Live">Live</button>
   <button id=micbtn class=iconbtn onclick=micToggle() title="Push-to-talk (browser STT)">PTT</button>
   <button id=spkbtn class=iconbtn onclick=voiceToggle() title="Speak replies + keep listening">TTS</button>
   <button id=vfbtn class=iconbtn onclick=vfToggle() title="ai-coustics Voice Focus — clean the mic audio">VF</button>
   <input id=inp placeholder="Speak or type — e.g. 'point at the laptop'" autofocus>
   <button class=send onclick=send()>Send</button></div>
</div>
</main>
<script>
const cam=document.getElementById('cam'),ov=document.getElementById('overlay'),log=document.getElementById('log'),inp=document.getElementById('inp');
const statusEl=document.getElementById('status');
function setStatus(cls,txt){statusEl.className='badge'+(cls?' '+cls:'');statusEl.innerHTML='<span class=dot></span>'+txt;}
(async()=>{try{const h=await(await fetch('/api/health')).json();setStatus(h.motor_frame?'':'err',h.motor_frame?('Connected · '+(h.bus||'arm')):'No robot');}catch(e){setStatus('err','Offline');}})();
let previewSrc='merged';
function refresh(){cam.src='/api/frame.jpg?src='+previewSrc+'&t='+Date.now();}
function setView(s){previewSrc=s;['vboth','vtop','vwrist'].forEach(id=>{const b=document.getElementById(id);if(b)b.classList.toggle('on',({merged:'vboth','0':'vtop','1':'vwrist'})[s]===id);});fetch('/api/view/'+s,{method:'POST'}).catch(()=>{});refresh();}
async function loadCams(){try{const d=await(await fetch('/api/cameras')).json();const sel=document.getElementById('camsel');sel.innerHTML='';(d.cameras||[]).forEach(c=>{const o=document.createElement('option');const tag=c.slot===0?'[1] ':c.slot===1?'[2] ':'    ';o.value=c.queue;o.textContent=tag+c.queue.replace('usbvideo/','').slice(0,10)+(c.has_frame?' ✓':'');o.selected=c.queue===d.slot0;sel.appendChild(o);});}catch(e){add('act','!cameras: '+e);}}
async function pickCam(slot){const q=document.getElementById('camsel').value;if(!q)return;add('act','cam:cam → slot '+(slot+1)+' ('+q.slice(-6)+')');try{await fetch('/api/setcam',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({slot:slot,queue:q})});setTimeout(refresh,400);setTimeout(loadCams,700);}catch(e){add('act','!'+e);}}
cam.onload=()=>setTimeout(refresh,250); cam.onerror=()=>setTimeout(refresh,1000);
setView('0'); loadCams();   // default to the single primary camera (most setups have one)
// --- Teach mode: limp the arm, point it at a spot, click that spot to save a reference pose ---
let teachMode=false;
function teachToggle(){teachMode=!teachMode;document.getElementById('teachbtn').classList.toggle('on',teachMode);cam.style.cursor=teachMode?'crosshair':'';add('act',teachMode?'teach mode ON — limp the arm (Torque off), hand-point it at a spot, then CLICK that spot on the camera':'teach mode off');if(teachMode)drawRefs();}
async function drawRefs(){try{const d=await(await fetch('/api/calib')).json();ov.innerHTML='';const rc=cam.getBoundingClientRect(),sc=ov.getBoundingClientRect();(d.refs||[]).forEach((r,i)=>{const[rx,ry]=r.point;const dot=document.createElement('div');dot.className='pt';dot.style.background='#F4B942';dot.style.left=((rc.left-sc.left)+rx/1000*rc.width)+'px';dot.style.top=((rc.top-sc.top)+ry/1000*rc.height)+'px';const l=document.createElement('div');l.className='lbl';l.style.background='#F4B942';l.style.color='#1a1505';l.textContent='ref '+(i+1);dot.appendChild(l);ov.appendChild(dot);});}catch(e){}}
cam.addEventListener('click',async(e)=>{if(!teachMode)return;const r=cam.getBoundingClientRect();const x=Math.round((e.clientX-r.left)/r.width*1000),y=Math.round((e.clientY-r.top)/r.height*1000);try{const d=await(await fetch('/api/calib',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({x,y})})).json();add('act','taught ref '+d.count+' at ['+x+','+y+']');drawRefs();}catch(err){add('act','! teach: '+err);}});
async function clearRefs(){await fetch('/api/calib/clear',{method:'POST'});ov.innerHTML='';add('act','cleared taught references');}
function add(c,txt){const d=document.createElement('div');d.className='msg '+c;d.textContent=txt;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
function draw(points){ov.innerHTML='';const r=cam.getBoundingClientRect(),s=ov.getBoundingClientRect();
 const ox=r.left-s.left,oy=r.top-s.top;
 (points||[]).forEach(p=>{if(!p.point)return;const[y,x]=p.point;
   const px=ox+x/1000*r.width,py=oy+y/1000*r.height;
   const dot=document.createElement('div');dot.className='pt';dot.style.left=px+'px';dot.style.top=py+'px';
   const l=document.createElement('div');l.className='lbl';l.textContent=p.label||'';dot.appendChild(l);ov.appendChild(dot);});
 setTimeout(()=>ov.innerHTML='',6000);}
async function send(){const m=inp.value.trim();if(!m)return;inp.value='';add('me',m);const w=add('bot','…');
 try{const r=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:m})});
   const d=await r.json();w.textContent=d.reply||'(no reply)';if(d.action)add('act',''+d.action);draw(d.points);
   if(voiceMode)speak(d.reply);}
 catch(e){w.textContent='Error: '+e;}}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
// --- Voice: speech-to-text (mic) + text-to-speech (replies) ---
const micBtn=document.getElementById('micbtn'),spkBtn=document.getElementById('spkbtn');
let recog=null,listening=false,voiceMode=false;
function micToggle(){
 const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!SR){add('act','!no speech recognition in this browser — use Chrome or Edge');return;}
 if(listening){recog.stop();return;}
 recog=new SR();recog.lang='en-US';recog.interimResults=true;recog.continuous=false;
 recog.onstart=()=>{listening=true;micBtn.classList.add('on');};
 recog.onend=()=>{listening=false;micBtn.classList.remove('on');};
 recog.onerror=e=>{listening=false;micBtn.classList.remove('on');if(e.error!=='no-speech'&&e.error!=='aborted')add('act','!mic: '+e.error);};
 recog.onresult=e=>{let t='';for(const r of e.results)t+=r[0].transcript;inp.value=t;
   if(e.results[e.results.length-1].isFinal){recog.stop();send();}};
 recog.start();}
function speak(text){
 if(!text||!window.speechSynthesis)return;
 speechSynthesis.cancel();
 const u=new SpeechSynthesisUtterance(text);u.lang='en-US';u.rate=1.05;
 u.onend=()=>{if(voiceMode&&!listening)setTimeout(micToggle,300);};  // hands-free: listen again
 speechSynthesis.speak(u);}
function voiceToggle(){
 voiceMode=!voiceMode;spkBtn.classList.toggle('on',voiceMode);
 add('act','voice mode '+(voiceMode?'ON — talk, it replies aloud and keeps listening':'OFF'));
 if(voiceMode&&!listening)micToggle();}
async function t(v){await fetch('/api/torque/'+v,{method:'POST'});add('act','torque '+(v?'ON':'OFF'));}
async function rec(){const n=prompt('Pose name (e.g. left, center, right):');if(!n)return;
 await fetch('/api/record/'+encodeURIComponent(n),{method:'POST'});add('act','recorded pose "'+n+'"');}
async function callTool(n,a){add('act',''+n);try{const r=await(await fetch('/api/tool',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:n,args:a||{}})})).json();add('act','   → '+r.result);}catch(e){add('act','!'+e);}}
// --- ER 1.6 detection overlay on the live preview ---
const detBtn=document.getElementById('detbtn'),detov=document.getElementById('detov');
let detOn=false,detTimer=null;
function drawBoxes(boxes){
 const r=cam.getBoundingClientRect(),s=detov.getBoundingClientRect(),ox=r.left-s.left,oy=r.top-s.top;
 detov.innerHTML='';
 (boxes||[]).forEach(b=>{const[ymin,xmin,ymax,xmax]=b.box;
   const d=document.createElement('div');d.className='dbox';
   d.style.left=(ox+xmin/1000*r.width)+'px';d.style.top=(oy+ymin/1000*r.height)+'px';
   d.style.width=((xmax-xmin)/1000*r.width)+'px';d.style.height=((ymax-ymin)/1000*r.height)+'px';
   const l=document.createElement('div');l.className='dlbl';l.textContent=b.label||'';d.appendChild(l);
   detov.appendChild(d);});
}
async function detTick(){if(!detOn)return;try{const d=await (await fetch('/api/detect')).json();if(detOn)drawBoxes(d.boxes);}catch(e){}}
function detToggle(){
 detOn=!detOn;detBtn.classList.toggle('on',detOn);
 if(detOn){add('act','ER 1.6 overlay on');detTick();detTimer=setInterval(detTick,2200);}
 else{if(detTimer)clearInterval(detTimer);detov.innerHTML='';add('act','overlay off');}
}
// --- Gemini Live (raw WebSocket): native bidirectional voice + video + tools ---
const liveBtn=document.getElementById('livebtn');
let lws=null,liveOn=false,micCtx=null,micProc=null,micStream=null,playCtx=null,playTime=0,frameTimer=null,curBot=null,curUser=null;
function b64FromBytes(u8){let s='';const CH=0x8000;for(let i=0;i<u8.length;i+=CH)s+=String.fromCharCode.apply(null,u8.subarray(i,i+CH));return btoa(s);}
function playPCM(b64){
 const raw=atob(b64),u8=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)u8[i]=raw.charCodeAt(i);
 const dv=new DataView(u8.buffer),n=u8.length>>1,f=new Float32Array(n);
 for(let i=0;i<n;i++)f[i]=dv.getInt16(i*2,true)/32768;
 const b=playCtx.createBuffer(1,n,24000);b.getChannelData(0).set(f);
 const s=playCtx.createBufferSource();s.buffer=b;s.connect(playCtx.destination);
 const now=playCtx.currentTime;if(playTime<now)playTime=now;s.start(playTime);playTime+=b.duration;}
async function liveToggle(){
 if(liveOn){stopLive();return;}
 liveOn=true;liveBtn.classList.add('on');add('act','connecting Live…');
 let cfg;try{cfg=await (await fetch('/api/live-config')).json();}catch(e){add('act','!config: '+e);return stopLive();}
 playCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:24000});
 lws=new WebSocket('wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key='+encodeURIComponent(cfg.apiKey));
 lws.onopen=()=>lws.send(JSON.stringify({setup:{model:'models/'+cfg.model,generationConfig:{responseModalities:['AUDIO']},systemInstruction:{parts:[{text:cfg.system}]},tools:[{functionDeclarations:cfg.decls}],inputAudioTranscription:{},outputAudioTranscription:{}}}));
 lws.onmessage=async(ev)=>{
   let d=ev.data;if(d instanceof Blob)d=await d.text();const m=JSON.parse(d);
   if(m.setupComplete){setStatus('live','Live · listening');add('act','live — talk now (headphones recommended). 3.1 sees the camera and drives the arm.');startMic();startFrames();return;}
   const sc=m.serverContent;
   if(sc){
     if(sc.modelTurn&&sc.modelTurn.parts)for(const p of sc.modelTurn.parts){if(p.inlineData&&p.inlineData.data)playPCM(p.inlineData.data);}
     if(sc.inputTranscription&&sc.inputTranscription.text){if(!curUser)curUser=add('me','');curUser.textContent+=sc.inputTranscription.text;}
     if(sc.outputTranscription&&sc.outputTranscription.text){if(!curBot)curBot=add('bot','');curBot.textContent+=sc.outputTranscription.text;}
     if(sc.turnComplete){curBot=null;curUser=null;}
   }
   if(m.toolCall){
     const fr=[];
     for(const fc of m.toolCall.functionCalls){
       add('act',''+fc.name+' '+JSON.stringify(fc.args||{}));
       let result='ok';
       try{result=(await (await fetch('/api/tool',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:fc.name,args:fc.args||{}})})).json()).result;}catch(e){result='error';}
       add('act','   → '+result);
       fr.push({id:fc.id,name:fc.name,response:{result}});
     }
     lws.send(JSON.stringify({toolResponse:{functionResponses:fr}}));
   }
 };
 lws.onerror=()=>add('act','!live socket error');
 lws.onclose=()=>{if(liveOn)add('act','live closed');};
}
async function startMic(){
 micStream=await navigator.mediaDevices.getUserMedia({audio:true});
 micCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
 const src=micCtx.createMediaStreamSource(micStream);micProc=micCtx.createScriptProcessor(4096,1,1);
 const mute=micCtx.createGain();mute.gain.value=0;src.connect(micProc);micProc.connect(mute);mute.connect(micCtx.destination);
 micProc.onaudioprocess=(e)=>{
   if(!lws||lws.readyState!==1)return;
   const f=e.inputBuffer.getChannelData(0);
   const vf=window._vf;
   if(vf&&vf.on&&vf.proc){            // ai-coustics  enhance in fixed n-frame blocks
     const merged=new Float32Array(_vfPending.length+f.length);merged.set(_vfPending,0);merged.set(f,_vfPending.length);
     let off=0;
     while(merged.length-off>=vf.n){const chunk=merged.slice(off,off+vf.n);try{vf.proc.processInterleaved(chunk);}catch(_){};sendF32(chunk);off+=vf.n;}
     _vfPending=merged.slice(off);
   } else { sendF32(f); }
 };
}
let _vfPending=new Float32Array(0);
function sendF32(f){
 const i16=new Int16Array(f.length);
 for(let i=0;i<f.length;i++){let s=Math.max(-1,Math.min(1,f[i]));i16[i]=s<0?s*32768:s*32767;}
 lws.send(JSON.stringify({realtimeInput:{audio:{data:b64FromBytes(new Uint8Array(i16.buffer)),mimeType:'audio/pcm;rate=16000'}}}));
}
function startFrames(){
 frameTimer=setInterval(async()=>{
   if(!lws||lws.readyState!==1)return;
   try{const ab=await (await fetch('/api/frame.jpg?src=merged&t='+Date.now())).arrayBuffer();  // both cameras -> 3.1
     lws.send(JSON.stringify({realtimeInput:{video:{data:b64FromBytes(new Uint8Array(ab)),mimeType:'image/jpeg'}}}));}catch(e){}
 },1000);   // ~1 FPS so 3.1 sees the result of each move
}
function stopLive(){
 liveOn=false;liveBtn.classList.remove('on');setStatus('','Connected');add('act','live off');
 try{if(frameTimer)clearInterval(frameTimer);}catch(e){}
 try{if(micProc)micProc.disconnect();}catch(e){}
 try{if(micStream)micStream.getTracks().forEach(t=>t.stop());}catch(e){}
 try{if(micCtx)micCtx.close();}catch(e){}
 try{if(lws)lws.close();}catch(e){}
 lws=null;curBot=null;curUser=null;
}
</script>
<script type="module">
// ai-coustics Voice Focus (WASM) — cleans the mic audio in-browser before it streams to Gemini Live.
window.vfToggle=async function(){
 const btn=document.getElementById('vfbtn');
 if(window._vf&&window._vf.on){window._vf.on=false;btn.classList.remove('on');add('act','Voice Focus OFF');return;}
 try{
   if(!window._vf){
     add('act','loading Voice Focus…');
     const lib=await import('https://cdn.jsdelivr.net/npm/@ai-coustics/aic-sdk-wasm@0.20.0/aic_sdk_wasm.js');
     await lib.default();
     const cfg=await (await fetch('/api/live-config')).json();
     if(!cfg.aicLicense){add('act','!no ai-coustics license (set AIC_SDK_LICENSE in .env)');return;}
     const bytes=new Uint8Array(await (await fetch(cfg.aicModelUrl)).arrayBuffer());
     const model=lib.Model.fromBytes(bytes);
     const sr=model.getOptimalSampleRate(), n=model.getOptimalNumFrames(sr);
     const proc=new lib.Processor(model, cfg.aicLicense);
     proc.initialize(sr,1,n,false);
     window._vf={proc,n,sr,on:true};
     add('act','Voice Focus ON ('+sr+'Hz · '+n+'-frame blocks)');
   } else { window._vf.on=true; add('act','Voice Focus ON'); }
   btn.classList.add('on');
 }catch(e){ add('act','!Voice Focus failed: '+e); }
};
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.getenv("STATION_SERVER", "localhost"))
    p.add_argument("--bus-serial", default=os.getenv("BUS_SERIAL", "auto"))
    p.add_argument("--video-queue", default=os.getenv("VIDEO_QUEUE", ""))
    p.add_argument("--video-queue2", default=os.getenv("VIDEO_QUEUE2", ""))   # wrist camera
    p.add_argument("--base-motor", type=int, default=int(os.getenv("BASE_MOTOR", "1")))
    p.add_argument("--tilt-motor", type=int, default=int(os.getenv("TILT_MOTOR", "0")))
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    S.server, S.want_bus, S.video_queue, S.base_motor = a.server, a.bus_serial, a.video_queue, a.base_motor
    S.video_queue2 = a.video_queue2
    S.tilt_motor = a.tilt_motor
    uvicorn.run(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
