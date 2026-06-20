"""Fuse Gemini Robotics-ER 1.6 with the NormaCore Station control server.

Pipeline:  station camera  ->  ER 1.6 (where is it?)  ->  arm pose.

  station daemon  --(USB)-->  server.py (:8000)  --HTTP-->  this script  --HTTPS-->  Gemini ER 1.6

Run order:
  1. station --web --tcp                       # the robot daemon (their binary)
  2. python server.py --bus-serial ...         # the control HTTP API (sibling file)
  3. export GEMINI_API_KEY=...
     pip install google-genai pillow requests
     python bridge.py "red block"              # point the arm at the red block
     python bridge.py --look                    # just say what's on the table

point_arm_at() / describe_table() below are written so the Gemini Live voice
loop can register them as tools unchanged — voice is the only thing left to add.
"""

from __future__ import annotations

import argparse
import io
import json
import os

import requests
from PIL import Image
from google import genai
from google.genai import types

CONTROL = os.environ.get("CONTROL_URL", "http://localhost:8000")
MODEL_ID = "gemini-robotics-er-1.6-preview"
client = genai.Client()  # reads GEMINI_API_KEY


# --- reused from the ER quickstart Colab ------------------------------------
def parse_json(json_output: str) -> str:
    """Strip ```json fencing the model sometimes adds (verbatim from the Colab)."""
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "```json":
            json_output = "\n".join(lines[i + 1:]).split("```")[0]
            break
    return json_output


def call_er(img: Image.Image, prompt: str, thinking_budget: int = 0) -> str:
    """The Colab's call_gemini_robotics_er, minus the notebook-only bits."""
    cfg = types.GenerateContentConfig(
        temperature=0.5,
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
    )
    resp = client.models.generate_content(model=MODEL_ID, contents=[img, prompt], config=cfg)
    return parse_json(resp.text)


# --- Station bridge ---------------------------------------------------------
def get_frame() -> Image.Image:
    r = requests.get(f"{CONTROL}/frame.jpg", timeout=5)
    r.raise_for_status()
    return Image.open(io.BytesIO(r.content)).convert("RGB")


def replay(zone: str) -> dict:
    r = requests.post(f"{CONTROL}/replay_pose", json={"name": zone}, timeout=10)
    r.raise_for_status()
    return r.json()


# --- the two functions Gemini Live will call as tools -----------------------
def describe_table() -> list[dict]:
    """List what's on the table (the 'what do you see' use case)."""
    prompt = ('Point to no more than 10 items on the table. The label is an '
              'identifying name. JSON: [{"point":[y,x],"label":<label>}, ...]. '
              'Points are [y, x] normalized 0-1000.')
    return json.loads(call_er(get_frame(), prompt))


def point_arm_at(target: str) -> str | None:
    """Find `target` with ER, bucket its x into a zone, replay that saved pose."""
    prompt = (f'Point to the {target}. JSON: [{{"point":[y,x],"label":<label>}}]. '
              'Points are [y, x] normalized 0-1000. If absent, return [].')
    pts = json.loads(call_er(get_frame(), prompt))
    if not pts:
        print(f"ER couldn't find '{target}'.")
        return None
    y, x = pts[0]["point"]
    zone = "left" if x < 333 else "center" if x < 666 else "right"
    print(f"'{target}' at [y={y}, x={x}] -> zone '{zone}'")
    replay(zone)
    return zone


def main():
    ap = argparse.ArgumentParser(description="ER 1.6 -> NormaCore arm")
    ap.add_argument("target", nargs="?", help="object to point the arm at")
    ap.add_argument("--look", action="store_true", help="just list what's on the table")
    args = ap.parse_args()

    if args.look or not args.target:
        for o in describe_table():
            print(f"  {o.get('label', '?'):<22} {o.get('point')}")
    else:
        point_arm_at(args.target)


if __name__ == "__main__":
    main()
