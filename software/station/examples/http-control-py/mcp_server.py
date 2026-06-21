"""MCP server — integrates the NormaCore station API into Claude (or Codex) so the
agent can CONTROL THE ROBOT BY LOOKING THROUGH THE CAMERAS.

This is the hackathon-track deliverable: Claude is the controlling brain. It calls
`look` to SEE the camera (image returned straight to Claude's vision), reasons about
the scene, then calls `point_at` / `grab` / `move_joint` to ACT on the real arm.

It's a thin client over the chat.py robot HTTP API (which holds the station
connection), so it reuses 100% of the station integration + ER 1.6 vision.

Setup (when the arm + cameras are up):
    1. start the robot API:        uv run python chat.py        # serves :8000
    2. register with Claude Code:  claude mcp add norma -- uv run python mcp_server.py
       (or add it to claude_desktop_config.json / a Codex MCP config)
    3. ask Claude: "Look at the table and grab the red block."

Env: ROBOT_API (default http://localhost:8000)
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP, Image

BASE = os.getenv("ROBOT_API", "http://localhost:8000")
mcp = FastMCP("norma-arm")
_http = httpx.Client(base_url=BASE, timeout=90.0)


@mcp.tool()
def look() -> Image:
    """Look through the robot's camera(s) and return the current view.

    Returns the live image — overhead + wrist merged into one frame when both
    cameras are present. Call this first to SEE the scene before deciding what to do.
    """
    r = _http.get("/api/frame.jpg")
    r.raise_for_status()
    return Image(data=r.content, format="jpeg")


@mcp.tool()
def detect_objects() -> list:
    """Detect objects in view using Gemini ER 1.6. Returns a list of
    {box: [ymin,xmin,ymax,xmax], label} with coordinates normalized 0-1000."""
    return _http.get("/api/detect").json().get("boxes", [])


@mcp.tool()
def arm_state() -> dict:
    """Current joint positions (normalized 0..1) + calibrated ranges + which motor is the gripper."""
    return _http.get("/api/state").json()


@mcp.tool()
def point_at(target: str) -> str:
    """Start aiming the arm at a named object (closed-loop visual servo). Returns immediately
    and keeps moving in the background — call look() again to verify, or retarget/stop to correct."""
    return _http.post("/api/tool", json={"name": "point_at", "args": {"target": target}}).json()["result"]


@mcp.tool()
def retarget(target: str) -> str:
    """While the arm is already moving, switch what it aims at (e.g. 'no, the red one')."""
    return _http.post("/api/tool", json={"name": "retarget", "args": {"target": target}}).json()["result"]


@mcp.tool()
def nudge(direction: str) -> str:
    """Nudge the current aim a bit: 'left', 'right', 'up', or 'down'."""
    return _http.post("/api/tool", json={"name": "nudge", "args": {"direction": direction}}).json()["result"]


@mcp.tool()
def stop() -> str:
    """Stop the arm's current motion immediately."""
    return _http.post("/api/tool", json={"name": "stop", "args": {}}).json()["result"]


@mcp.tool()
def grab(target: str) -> str:
    """Pick up a named object: open the claw, reach toward it, close the claw."""
    return _http.post("/api/tool", json={"name": "grab", "args": {"target": target}}).json()["result"]


@mcp.tool()
def move_joint(motor_id: int, normalized: float) -> str:
    """Move one joint to a normalized position (0 = range min, 1 = range max)."""
    return _http.post("/api/tool", json={"name": "move_joint",
                                         "args": {"motor_id": motor_id, "normalized": normalized}}).json()["result"]


@mcp.tool()
def open_gripper() -> str:
    """Open the claw/gripper."""
    return _http.post("/api/tool", json={"name": "open_gripper", "args": {}}).json()["result"]


@mcp.tool()
def close_gripper() -> str:
    """Close the claw/gripper to hold an object."""
    return _http.post("/api/tool", json={"name": "close_gripper", "args": {}}).json()["result"]


@mcp.tool()
def set_torque(enable: bool) -> str:
    """Enable (stiffen) or disable (limp) the arm motors."""
    return _http.post("/api/tool", json={"name": "set_torque", "args": {"enable": enable}}).json()["result"]


if __name__ == "__main__":
    mcp.run()  # stdio transport for Claude Desktop / Claude Code / Codex
