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
from fastapi.responses import HTMLResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402
from PIL import Image  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
import uvicorn  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("chat")

MODEL_ID = "gemini-robotics-er-1.6-preview"
LOCATE_MODEL = os.getenv("LOCATE_MODEL", "gemini-2.5-flash")  # fast model used inside the servo loop
SERVO_MODEL = os.getenv("SERVO_MODEL", "gemini-robotics-er-1.6-preview")  # ER 1.6 = the vision brain
SERVO_TOL = 70           # off-center distance (0-1000) = "aimed at it"
CENTER_OK = 150          # "roughly in the middle" — center to here before reaching
SERVO_MAX_ITERS = 28     # longer loop
SERVO_GAIN = 0.08        # smaller = slower, smoother (was 0.13)
SERVO_MAX_STEP = 0.06    # cap on a single iteration's joint move (normalized)
SERVO_SETTLE = 0.7       # seconds to let the servo move + camera refresh between checks
REACH_WEIGHT = 0.7       # how much "get closer" (grow on-screen size) matters vs centering
REACH_SIZE_TARGET = 0.18 # stop once the target fills ~18% of the frame (close enough)
GRIPPER_OPEN = float(os.getenv("GRIPPER_OPEN", "0.0"))   # normalized claw-open position (flip if reversed)
GRIPPER_CLOSE = float(os.getenv("GRIPPER_CLOSE", "1.0"))  # normalized claw-closed position
# Gemini Live (browser WebSocket voice agent)
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-3.1-flash-live-preview")
LIVE_SYSTEM = (
    "You are the voice of a NormaCore robot arm. You do NOT see the camera directly — the robot's vision is a "
    "separate model (Gemini ER 1.6). To see the scene, call look(). To aim the arm at something, call "
    "point_at(target) — it visually servos the arm until the object is centered. To pick something up call "
    "grab(target) (opens the claw, reaches, closes); open_gripper/close_gripper control the claw directly. Use "
    "move_joint for direct joint moves, replay_pose for saved poses, set_torque to limp/stiffen the arm. "
    "Keep spoken replies short and conversational; narrate what you're doing as you move."
)
FUNC_DECLS = [
    {"name": "look", "description": "Look through the robot's camera (Gemini ER 1.6) and describe what is visible. Call whenever the user asks what you see or to find/identify something.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "point_at", "description": "Aim the camera/arm at a named object via closed-loop visual servoing until centered.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}},
    {"name": "move_joint", "description": "Move one joint to a normalized position (0=range min, 1=range max).",
     "parameters": {"type": "object", "properties": {"motor_id": {"type": "integer"}, "normalized": {"type": "number"}}, "required": ["motor_id", "normalized"]}},
    {"name": "replay_pose", "description": "Move the arm to a saved pose by name.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "set_torque", "description": "Enable (stiffen) or disable (limp) the arm motors.",
     "parameters": {"type": "object", "properties": {"enable": {"type": "boolean"}}, "required": ["enable"]}},
    {"name": "open_gripper", "description": "Open the claw/gripper.", "parameters": {"type": "object", "properties": {}}},
    {"name": "close_gripper", "description": "Close the claw/gripper to hold an object.", "parameters": {"type": "object", "properties": {}}},
    {"name": "grab", "description": "Pick up an object: open the claw, aim/reach toward the named object, then close the claw.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string"}}, "required": ["target"]}},
]
MOTOR_QUEUE = "st3215/inference"
RAM_TORQUE_ENABLE, RAM_GOAL_POSITION, RAM_PRESENT_POSITION = 0x28, 0x2A, 0x38
MAX_STEP, SIGN_BIT = 4095, 0x8000
POSES_PATH = _HERE.parent / "poses.json"


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
    video_queue = ""
    jpeg: bytes | None = None
    gemini = None


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


async def _follow_video():
    # The usbvideo queue carries RxEnvelope messages; JPEG frames live inside
    # envelopes of type ET_FRAMES (other types are device-connect/error events).
    from target.gen_python.protobuf.drivers.usbvideo.usbvideo import (
        RxEnvelopeReader, RxEnvelopeType)
    q: asyncio.Queue = asyncio.Queue()
    S.client.follow(S.video_queue, q)
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
                S.jpeg = bytes(frames[0])
        except Exception:
            logger.exception("video decode failed")


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


def _frame_pil() -> Image.Image:
    if S.jpeg is not None:
        return Image.open(io.BytesIO(S.jpeg)).convert("RGB")
    # webcam fallback
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
    raise HTTPException(503, "no camera frame (no station video queue and no webcam).")


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


async def _locate_box(target: str):
    """Find `target` -> (cx, cy, size_fraction) in 0-1000 / 0..1, or None.

    size_fraction (bounding-box area / frame) is the distance proxy: bigger = closer.
    """
    img = _frame_pil()
    prompt = (f'Find the {target}. Return ONLY JSON: a bounding box [{{"box_2d":[ymin,xmin,ymax,xmax]}}] '
              f'if you can see it clearly, otherwise a point [{{"point":[y,x]}}]. Normalized 0-1000. '
              f'If absent, return [].')
    cfg = types.GenerateContentConfig(temperature=0.0,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    try:
        resp = await asyncio.to_thread(S.gemini.models.generate_content,
                                       model=SERVO_MODEL, contents=[img, prompt], config=cfg)
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


async def _set_gripper(open_it: bool) -> str:
    await _torque(True)
    g = _gripper_id()
    await _move_norm({g: GRIPPER_OPEN if open_it else GRIPPER_CLOSE})
    return f"claw {'opened' if open_it else 'closed'} (M{g})"


async def _servo_to(target: str) -> str:
    """Two-brain visual servo: ER 1.6 is the vision brain, the arm is driven slowly.

    Phase 1 (center): move joints to bring `target` to the MIDDLE of the frame.
    Phase 2 (reach):  only once it's roughly centered, extend toward it (grow its size).
    Each step ER 1.6 locates the target; we nudge one joint a small, shrinking step,
    then verify on the next frame: worse -> step back + reverse; no effect -> freeze
    that joint (this is what kills the random drift); better -> keep its direction.
    """
    active = _servo_joints()
    if not active:
        return "no servo joints available"
    sign = {j: 1 for j in active}
    gain = {j: SERVO_GAIN for j in active}
    frozen: set[int] = set()
    prev_pos: dict[int, float] = {}
    prev_cost = None
    last_j = None
    phase = "center"
    qi = 0
    await _torque(True)
    log = [f"two-brain servo (ER 1.6 vision) joints {active} -> '{target}'"]
    for i in range(SERVO_MAX_ITERS):
        box = await _locate_box(target)
        if box is None:
            log.append(f"{i}: target not in view")
            continue
        cx, cy, size = box
        have_size = size is not None
        cdist = math.hypot(cx - 500.0, cy - 500.0)
        # phase transitions with hysteresis (reach only when ER gives a box = a size)
        if phase == "center" and cdist < CENTER_OK and have_size:
            phase = "reach"; frozen.clear(); gain = {j: SERVO_GAIN for j in active}; prev_cost = None; last_j = None
            log.append("  in the middle — reaching")
        elif phase == "reach" and cdist > CENTER_OK * 1.6:
            phase = "center"; frozen.clear(); gain = {j: SERVO_GAIN for j in active}; prev_cost = None; last_j = None
            log.append("  drifted off-center — re-centering")
        cost = cdist / 707.0 + (REACH_WEIGHT * max(0.0, REACH_SIZE_TARGET - size) / REACH_SIZE_TARGET
                                if (phase == "reach" and have_size) else 0.0)
        log.append(f"{i}: c=({int(cx)},{int(cy)}) off={int(cdist)} size={(size if have_size else -1):.2f} [{phase}]")
        if cdist < SERVO_TOL and (not have_size or size >= REACH_SIZE_TARGET):
            log.append("aimed ✓" if not have_size else "aimed + reached ✓")
            break
        # verify the previous move
        if last_j is not None and prev_cost is not None:
            dcost = cost - prev_cost
            if dcost > 0.01:                       # worse -> step back + reverse + damp
                sign[last_j] *= -1
                gain[last_j] = max(0.03, gain[last_j] * 0.6)
                await _move_norm({last_j: prev_pos[last_j]})
            elif abs(dcost) < 0.01:                # no measurable effect -> freeze it
                gain[last_j] *= 0.5
                if gain[last_j] < 0.035:
                    frozen.add(last_j)
                    log.append(f"  joint {last_j}: no effect, frozen")
        prev_cost = cost
        avail = [j for j in active if j not in frozen]
        if not avail:
            log.append("converged (no joint improves it further)")
            break
        # nudge the next available joint; step shrinks as we get closer (slow + precise)
        j = avail[qi % len(avail)]; qi += 1
        cur = _norm_of(j)
        prev_pos[j] = cur
        last_j = j
        scale = max(0.35, min(1.0, cost))
        d = max(-SERVO_MAX_STEP, min(SERVO_MAX_STEP, sign[j] * gain[j] * scale))
        await _move_norm({j: max(0.0, min(1.0, cur + d))})
        await asyncio.sleep(SERVO_SETTLE)
    return " | ".join(log[-18:])


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
        if name == "point_at":
            return await _servo_to((args.get("target") or "").strip())
        if name == "move_joint":
            await _torque(True)
            return f"moved joint {args['motor_id']} -> {await _move_norm({int(args['motor_id']): float(args['normalized'])})}"
        if name == "replay_pose":
            await _torque(True)
            return json.dumps(await _replay(args["name"]))
        if name == "set_torque":
            await _torque(bool(args["enable"]))
            return "torque " + ("enabled" if args["enable"] else "disabled (limp)")
        if name == "open_gripper":
            return await _set_gripper(True)
        if name == "close_gripper":
            return await _set_gripper(False)
        if name == "grab":
            t = (args.get("target") or "").strip()
            if not t:
                return "no target to grab"
            await _set_gripper(True)        # open the claw
            servo_log = await _servo_to(t)  # aim + reach toward it
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
        asyncio.create_task(_follow_video())
    else:
        logger.warning("no station video queue found — will try local webcam for the preview")
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
async def frame():
    pil = _frame_pil()
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    return Response(content=buf.getvalue(), media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


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
            "system": LIVE_SYSTEM, "decls": FUNC_DECLS}


@app.post("/api/tool")
async def api_tool(req: ToolReq):
    return {"result": await _run_tool(req.name, req.args)}


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


HTML = """<!doctype html><html><head><meta charset=utf-8><title>Norma Core · ER 1.6</title>
<style>
 :root{color-scheme:dark}
 *{box-sizing:border-box} body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;
   background:#0d0f12;color:#e7e9ec;height:100vh;display:flex}
 #left{flex:1;padding:18px;display:flex;flex-direction:column;gap:12px;min-width:0}
 #stage{position:relative;background:#000;border-radius:12px;overflow:hidden;flex:1;display:flex;align-items:center;justify-content:center}
 #cam{max-width:100%;max-height:100%;display:block}
 #overlay{position:absolute;inset:0;pointer-events:none}
 .pt{position:absolute;width:14px;height:14px;border-radius:50%;background:#2962FF;border:2px solid #fff;
   transform:translate(-50%,-50%);box-shadow:0 0 22px rgba(41,98,255,.7)}
 .lbl{position:absolute;background:#2962FF;color:#fff;font-size:12px;padding:2px 8px;border-radius:5px;
   transform:translate(12px,-10px);white-space:nowrap}
 #detov{position:absolute;inset:0;pointer-events:none}
 .dbox{position:absolute;border:2px solid #27e0a0;border-radius:4px;box-shadow:0 0 12px rgba(39,224,160,.35)}
 .dlbl{position:absolute;top:-17px;left:-2px;background:#27e0a0;color:#04140d;font-size:11px;font-weight:600;
   padding:1px 6px;border-radius:4px;white-space:nowrap}
 #detbtn.on{background:#15705a;border-color:#27e0a0;color:#d6fff0}
 #ctrls{display:flex;gap:8px;flex-wrap:wrap}
 button{background:#1b1f27;color:#e7e9ec;border:1px solid #2b303b;border-radius:8px;padding:8px 12px;cursor:pointer}
 button:hover{background:#262b35}
 #right{width:420px;border-left:1px solid #1b1f27;display:flex;flex-direction:column}
 #log{flex:1;overflow:auto;padding:18px;display:flex;flex-direction:column;gap:12px}
 .msg{padding:10px 13px;border-radius:12px;max-width:85%}
 .me{align-self:flex-end;background:#2962FF}
 .bot{align-self:flex-start;background:#1b1f27;border:1px solid #2b303b}
 .act{align-self:flex-start;font-size:12px;color:#7bdc9a;font-family:ui-monospace,monospace}
 #bar{display:flex;gap:8px;padding:14px;border-top:1px solid #1b1f27}
 #inp{flex:1;background:#11151b;border:1px solid #2b303b;border-radius:10px;color:#fff;padding:11px 13px;font:inherit}
 h3{margin:0;font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#8b93a1}
 #micbtn,#spkbtn,#livebtn{font-size:18px;line-height:1;width:44px}
 #micbtn.on{background:#c0392b;border-color:#e74c3c;animation:pulse 1.1s infinite}
 #spkbtn.on{background:#1e7d4f;border-color:#27ae60}
 #livebtn.on{background:#1e7d4f;border-color:#27ae60;animation:pulse 1.1s infinite}
 @keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(231,76,60,.55)}50%{box-shadow:0 0 0 9px rgba(231,76,60,0)}}
</style></head><body>
<div id=left>
 <h3>Live camera · arm view</h3>
 <div id=stage><img id=cam><div id=overlay></div><div id=detov></div></div>
 <div id=ctrls>
   <button id=detbtn onclick="detToggle()">👁 ER overlay</button>
   <button onclick="callTool('open_gripper')">✋ Open claw</button>
   <button onclick="callTool('close_gripper')">🤏 Close claw</button>
   <button onclick="t(0)">Torque OFF (limp)</button>
   <button onclick="t(1)">Torque ON</button>
   <button onclick="rec()">Record pose…</button>
 </div>
</div>
<div id=right>
 <div id=log><div class=msg bot>Hi! I can see through the arm's camera. Ask me what I see, or tell me to point at something.</div></div>
 <div id=bar>
   <button id=livebtn onclick=liveToggle() title="Native voice — Gemini Live over WebSocket">🎙</button>
   <button id=micbtn onclick=micToggle() title="Push-to-talk (browser speech → text)">🎤</button>
   <button id=spkbtn onclick=voiceToggle() title="Speak text replies + keep listening">🔊</button>
   <input id=inp placeholder="Speak or type — e.g. 'point at the laptop'" autofocus>
   <button onclick=send()>Send</button></div>
</div>
<script>
const cam=document.getElementById('cam'),ov=document.getElementById('overlay'),log=document.getElementById('log'),inp=document.getElementById('inp');
function refresh(){cam.src='/api/frame.jpg?t='+Date.now();}
cam.onload=()=>setTimeout(refresh,250); cam.onerror=()=>setTimeout(refresh,1000); refresh();
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
   const d=await r.json();w.textContent=d.reply||'(no reply)';if(d.action)add('act','⚙ '+d.action);draw(d.points);
   if(voiceMode)speak(d.reply);}
 catch(e){w.textContent='Error: '+e;}}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
// --- Voice: speech-to-text (mic) + text-to-speech (replies) ---
const micBtn=document.getElementById('micbtn'),spkBtn=document.getElementById('spkbtn');
let recog=null,listening=false,voiceMode=false;
function micToggle(){
 const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
 if(!SR){add('act','⚠ no speech recognition in this browser — use Chrome or Edge');return;}
 if(listening){recog.stop();return;}
 recog=new SR();recog.lang='en-US';recog.interimResults=true;recog.continuous=false;
 recog.onstart=()=>{listening=true;micBtn.classList.add('on');};
 recog.onend=()=>{listening=false;micBtn.classList.remove('on');};
 recog.onerror=e=>{listening=false;micBtn.classList.remove('on');if(e.error!=='no-speech'&&e.error!=='aborted')add('act','⚠ mic: '+e.error);};
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
 add('act','🔊 voice mode '+(voiceMode?'ON — talk, it replies aloud and keeps listening':'OFF'));
 if(voiceMode&&!listening)micToggle();}
async function t(v){await fetch('/api/torque/'+v,{method:'POST'});add('act','⚙ torque '+(v?'ON':'OFF'));}
async function rec(){const n=prompt('Pose name (e.g. left, center, right):');if(!n)return;
 await fetch('/api/record/'+encodeURIComponent(n),{method:'POST'});add('act','⚙ recorded pose "'+n+'"');}
async function callTool(n,a){add('act','⚙ '+n);try{const r=await(await fetch('/api/tool',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:n,args:a||{}})})).json();add('act','   → '+r.result);}catch(e){add('act','⚠ '+e);}}
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
 if(detOn){add('act','👁 ER 1.6 overlay on');detTick();detTimer=setInterval(detTick,2200);}
 else{if(detTimer)clearInterval(detTimer);detov.innerHTML='';add('act','👁 overlay off');}
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
 liveOn=true;liveBtn.classList.add('on');add('act','🎙 connecting Live…');
 let cfg;try{cfg=await (await fetch('/api/live-config')).json();}catch(e){add('act','⚠ config: '+e);return stopLive();}
 playCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:24000});
 lws=new WebSocket('wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key='+encodeURIComponent(cfg.apiKey));
 lws.onopen=()=>lws.send(JSON.stringify({setup:{model:'models/'+cfg.model,generationConfig:{responseModalities:['AUDIO']},systemInstruction:{parts:[{text:cfg.system}]},tools:[{functionDeclarations:cfg.decls}],inputAudioTranscription:{},outputAudioTranscription:{}}}));
 lws.onmessage=async(ev)=>{
   let d=ev.data;if(d instanceof Blob)d=await d.text();const m=JSON.parse(d);
   if(m.setupComplete){add('act','🎙 live — talk now (headphones recommended). Camera handled by ER 1.6.');startMic();return;}
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
       add('act','⚙ '+fc.name+' '+JSON.stringify(fc.args||{}));
       let result='ok';
       try{result=(await (await fetch('/api/tool',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({name:fc.name,args:fc.args||{}})})).json()).result;}catch(e){result='error';}
       add('act','   → '+result);
       fr.push({id:fc.id,name:fc.name,response:{result}});
     }
     lws.send(JSON.stringify({toolResponse:{functionResponses:fr}}));
   }
 };
 lws.onerror=()=>add('act','⚠ live socket error');
 lws.onclose=()=>{if(liveOn)add('act','live closed');};
}
async function startMic(){
 micStream=await navigator.mediaDevices.getUserMedia({audio:true});
 micCtx=new (window.AudioContext||window.webkitAudioContext)({sampleRate:16000});
 const src=micCtx.createMediaStreamSource(micStream);micProc=micCtx.createScriptProcessor(4096,1,1);
 const mute=micCtx.createGain();mute.gain.value=0;src.connect(micProc);micProc.connect(mute);mute.connect(micCtx.destination);
 micProc.onaudioprocess=(e)=>{
   if(!lws||lws.readyState!==1)return;
   const f=e.inputBuffer.getChannelData(0),i16=new Int16Array(f.length);
   for(let i=0;i<f.length;i++){let s=Math.max(-1,Math.min(1,f[i]));i16[i]=s<0?s*32768:s*32767;}
   lws.send(JSON.stringify({realtimeInput:{audio:{data:b64FromBytes(new Uint8Array(i16.buffer)),mimeType:'audio/pcm;rate=16000'}}}));
 };
}
function startFrames(){
 frameTimer=setInterval(async()=>{
   if(!lws||lws.readyState!==1)return;
   try{const ab=await (await fetch('/api/frame.jpg?t='+Date.now())).arrayBuffer();
     lws.send(JSON.stringify({realtimeInput:{video:{data:b64FromBytes(new Uint8Array(ab)),mimeType:'image/jpeg'}}}));}catch(e){}
 },1500);
}
function stopLive(){
 liveOn=false;liveBtn.classList.remove('on');add('act','🎙 live off');
 try{if(frameTimer)clearInterval(frameTimer);}catch(e){}
 try{if(micProc)micProc.disconnect();}catch(e){}
 try{if(micStream)micStream.getTracks().forEach(t=>t.stop());}catch(e){}
 try{if(micCtx)micCtx.close();}catch(e){}
 try{if(lws)lws.close();}catch(e){}
 lws=null;curBot=null;curUser=null;
}
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.getenv("STATION_SERVER", "localhost"))
    p.add_argument("--bus-serial", default=os.getenv("BUS_SERIAL", "auto"))
    p.add_argument("--video-queue", default=os.getenv("VIDEO_QUEUE", ""))
    p.add_argument("--base-motor", type=int, default=int(os.getenv("BASE_MOTOR", "1")))
    p.add_argument("--tilt-motor", type=int, default=int(os.getenv("TILT_MOTOR", "0")))
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    S.server, S.want_bus, S.video_queue, S.base_motor = a.server, a.bus_serial, a.video_queue, a.base_motor
    S.tilt_motor = a.tilt_motor
    uvicorn.run(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
