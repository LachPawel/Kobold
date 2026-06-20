"""Native-audio voice agent for the NormaCore / ElRobot arm via the Gemini Live API.

Talk to your laptop mic; Gemini sees the arm camera (video stream) and replies
in natural speech, calling arm tools — including the closed-loop visual-servo
"point at X". Reuses ALL the station/arm/camera/servo plumbing from chat.py.

Architecture: server-to-server, in-process. This script connects to the Live
API, streams your mic in + the arm camera in, plays Gemini's voice out, and
runs tool calls against the arm locally. The API key stays server-side.

Run:
    uv run python live.py
    # then just talk. Ctrl-C to stop.
    # macOS will ask for microphone permission the first time — allow it.

Watch the camera in the Station web UI (http://localhost:8889) or run chat.py
alongside for the preview.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import chat  # reuse station client, camera follow, arm control, servo loop  # noqa: E402
from chat import S  # noqa: E402
from station_py import new_station_client  # noqa: E402
from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
import sounddevice as sd  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s")
logger = logging.getLogger("live")

# Verified working for audio + tools + transcription on this account.
LIVE_MODEL = os.getenv("LIVE_MODEL", "gemini-2.5-flash-native-audio-preview-09-2025")
MIC_RATE, SPK_RATE, BLOCK = 16000, 24000, 1024

SYSTEM = (
    "You are the voice of a NormaCore robot arm with an eye-in-hand camera. "
    "You can SEE the live camera feed. Answer questions about what you see naturally and briefly. "
    "When the user asks you to point at / look at / find an object, call point_at with that object — "
    "it visually servos the arm until the object is centered. Use move_joint for direct joint moves "
    "(the gripper is the highest motor id), replay_pose for saved poses, set_torque to limp/stiffen the arm. "
    "Keep spoken replies short and conversational. Narrate what you're doing as you move."
)

TOOLS = [{"function_declarations": [
    {"name": "point_at",
     "description": "Aim the camera/arm at a named object using closed-loop visual servoing until it is centered in view.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "object to point at, e.g. 'the laptop'"}}, "required": ["target"]}},
    {"name": "move_joint",
     "description": "Move a single joint to a normalized position (0=range min, 1=range max).",
     "parameters": {"type": "object", "properties": {"motor_id": {"type": "integer"}, "normalized": {"type": "number"}}, "required": ["motor_id", "normalized"]}},
    {"name": "replay_pose",
     "description": "Move the arm to a previously saved pose by name.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "set_torque",
     "description": "Enable (stiffen) or disable (go limp) the arm motors.",
     "parameters": {"type": "object", "properties": {"enable": {"type": "boolean"}}, "required": ["enable"]}},
]}]


async def do_tool(name: str, args: dict) -> str:
    try:
        if name == "point_at":
            return await chat._servo_to((args.get("target") or "").strip())
        if name == "move_joint":
            await chat._torque(True)
            return f"moved joint {args['motor_id']} -> {await chat._move_norm({int(args['motor_id']): float(args['normalized'])})}"
        if name == "replay_pose":
            await chat._torque(True)
            return json.dumps(await chat._replay(args["name"]))
        if name == "set_torque":
            await chat._torque(bool(args["enable"]))
            return "torque " + ("enabled" if args["enable"] else "disabled (limp)")
    except Exception as e:
        logger.exception("tool %s failed", name)
        return f"error: {e}"
    return f"unknown tool {name}"


async def run():
    # --- bring up station + camera + arm (same config as chat.py / .env) ---
    S.server = os.getenv("STATION_SERVER", "localhost")
    S.want_bus = os.getenv("BUS_SERIAL", "auto")
    S.video_queue = os.getenv("VIDEO_QUEUE", "")
    S.base_motor = int(os.getenv("BASE_MOTOR", "1"))
    S.tilt_motor = int(os.getenv("TILT_MOTOR", "0"))
    S.client = await new_station_client(S.server, logger)
    S.gemini = genai.Client()
    asyncio.create_task(chat._follow_motors())
    if not S.video_queue:
        S.video_queue = chat._discover_video_queue()
    if S.video_queue:
        asyncio.create_task(chat._follow_video())
    await chat._wait_ready()
    print(f"arm ready: bus={S.bus_serial} | camera={'yes' if S.jpeg else 'no'} | model={LIVE_MODEL}")

    # --- local audio: mic in (16k) + speaker out (24k), raw int16 PCM ---
    loop = asyncio.get_running_loop()
    mic_q: asyncio.Queue = asyncio.Queue()

    def mic_cb(indata, frames, time_info, status):
        loop.call_soon_threadsafe(mic_q.put_nowait, bytes(indata))

    mic = sd.RawInputStream(samplerate=MIC_RATE, channels=1, dtype="int16", blocksize=BLOCK, callback=mic_cb)
    spk = sd.RawOutputStream(samplerate=SPK_RATE, channels=1, dtype="int16")
    mic.start()
    spk.start()

    config = {
        "response_modalities": ["AUDIO"],
        "system_instruction": SYSTEM,
        "tools": TOOLS,
        "input_audio_transcription": {},
        "output_audio_transcription": {},
    }

    print("\n🎙️  listening — just talk (Ctrl-C to stop)\n")
    async with S.gemini.aio.live.connect(model=LIVE_MODEL, config=config) as session:

        async def send_mic():
            while True:
                chunk = await mic_q.get()
                await session.send_realtime_input(audio=types.Blob(data=chunk, mime_type="audio/pcm;rate=16000"))

        async def send_video():
            while True:
                try:
                    buf = io.BytesIO()
                    chat._frame_pil().save(buf, format="JPEG")
                    await session.send_realtime_input(video=types.Blob(data=buf.getvalue(), mime_type="image/jpeg"))
                except Exception:
                    pass
                await asyncio.sleep(1.0)  # Live accepts <= 1 FPS

        async def receive():
            while True:
                async for resp in session.receive():
                    if resp.data:  # output audio (24k PCM) -> speaker
                        await asyncio.to_thread(spk.write, resp.data)
                    sc = resp.server_content
                    if sc and sc.input_transcription and sc.input_transcription.text:
                        print(f"🧑 {sc.input_transcription.text}")
                    if sc and sc.output_transcription and sc.output_transcription.text:
                        print(f"🤖 {sc.output_transcription.text}", flush=True)
                    if resp.tool_call:
                        replies = []
                        for fc in resp.tool_call.function_calls:
                            print(f"⚙  {fc.name}({dict(fc.args or {})})")
                            result = await do_tool(fc.name, dict(fc.args or {}))
                            print(f"   -> {result}")
                            replies.append(types.FunctionResponse(id=fc.id, name=fc.name, response={"result": result}))
                        await session.send_tool_response(function_responses=replies)

        try:
            await asyncio.gather(send_mic(), send_video(), receive())
        finally:
            mic.stop(); spk.stop()


def main():
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\nbye")


if __name__ == "__main__":
    main()
