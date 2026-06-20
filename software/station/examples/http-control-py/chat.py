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
import os
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
    base_motor = 1
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
        # map the target's x-coordinate to base-joint rotation (no calibration needed)
        x = None
        for p in points or []:
            if p.get("point"):
                x = p["point"][1]
                break
        if x is None:
            return "couldn't locate target to point at"
        norm = max(0.0, min(1.0, x / 1000.0))
        await _move_norm({S.base_motor: norm})
        return f"rotated base toward x={x} (norm {norm:.2f})"
    return ""


async def chat_turn(message: str) -> dict:
    await _wait_ready()
    img = _frame_pil()
    prompt = f"{_system_context()}\n\nUser: {message}"
    cfg = types.GenerateContentConfig(temperature=0.4,
                                      thinking_config=types.ThinkingConfig(thinking_budget=0))
    resp = await asyncio.to_thread(
        S.gemini.models.generate_content, model=MODEL_ID, contents=[img, prompt], config=cfg)
    try:
        data = json.loads(_parse_json(resp.text))
    except Exception:
        return {"reply": resp.text, "points": [], "action": ""}
    action_log = await _exec_arm(data.get("arm", {}), data.get("points", []))
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
    return await chat_turn(req.message)


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
</style></head><body>
<div id=left>
 <h3>Live camera · arm view</h3>
 <div id=stage><img id=cam><div id=overlay></div></div>
 <div id=ctrls>
   <button onclick="t(0)">Torque OFF (limp)</button>
   <button onclick="t(1)">Torque ON</button>
   <button onclick="rec()">Record pose…</button>
 </div>
</div>
<div id=right>
 <div id=log><div class=msg bot>Hi! I can see through the arm's camera. Ask me what I see, or tell me to point at something.</div></div>
 <div id=bar><input id=inp placeholder="Ask about the scene, or 'point at the…'" autofocus>
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
   const d=await r.json();w.textContent=d.reply||'(no reply)';if(d.action)add('act','⚙ '+d.action);draw(d.points);}
 catch(e){w.textContent='Error: '+e;}}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send();});
async function t(v){await fetch('/api/torque/'+v,{method:'POST'});add('act','⚙ torque '+(v?'ON':'OFF'));}
async function rec(){const n=prompt('Pose name (e.g. left, center, right):');if(!n)return;
 await fetch('/api/record/'+encodeURIComponent(n),{method:'POST'});add('act','⚙ recorded pose "'+n+'"');}
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=os.getenv("STATION_SERVER", "localhost"))
    p.add_argument("--bus-serial", default=os.getenv("BUS_SERIAL", "auto"))
    p.add_argument("--video-queue", default=os.getenv("VIDEO_QUEUE", ""))
    p.add_argument("--base-motor", type=int, default=int(os.getenv("BASE_MOTOR", "1")))
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    S.server, S.want_bus, S.video_queue, S.base_motor = a.server, a.bus_serial, a.video_queue, a.base_motor
    uvicorn.run(app, host="0.0.0.0", port=a.port)


if __name__ == "__main__":
    main()
