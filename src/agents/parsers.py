"""
Parsers for Agents
"""
import re
from agents.base import AgentOutput, CustomAction, CustomActionTypes


def _xy(s: str) -> tuple[float, float] | None:
    m = re.search(r'x=([\d.]+),\s*y=([\d.]+)', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'\(([\d.]+),\s*([\d.]+)', s)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None

def parse_pyautogui_action(raw_output: str) -> AgentOutput | None:
    """Parse a pyautogui / mobile tool-call string into AgentOutput. Returns None if unrecognised."""
    raw = {"content": raw_output}


    # ── pyautogui desktop actions ─────────────────────────────────────────
    if 'pyautogui.click' in raw_output or 'pyautogui.doubleClick' in raw_output:
        xy = _xy(raw_output)
        if xy:
            atype = "double_click" if "doubleClick" in raw_output else "click"
            return AgentOutput(coordinate=xy, action_type=atype, raw=raw)

    elif 'pyautogui.write' in raw_output:
        m = re.search(r"pyautogui\.write\(['\"](.+?)['\"]\)", raw_output)
        if m:
            return AgentOutput(text=m.group(1), action_type="write", raw=raw)

    elif 'pyautogui.press' in raw_output:
        m = re.search(r"pyautogui\.press\(['\"](.+?)['\"]\)", raw_output)
        if m:
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.KEY_PRESS, params={"key": m.group(1)}),
                action_type="key_press", raw=raw,
            )

    elif 'pyautogui.hotkey' in raw_output:
        m = re.search(r"pyautogui\.hotkey\((.+?)\)", raw_output)
        if m:
            keys = re.findall(r"['\"](.+?)['\"]", m.group(1))
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.HOTKEY, params={"keys": keys}),
                action_type="hotkey", raw=raw,
            )

    elif 'pyautogui.scroll' in raw_output:
        m = re.search(r"pyautogui\.scroll\((-?[\d.]+)\)", raw_output)
        if m:
            clicks = float(m.group(1))
            return AgentOutput(
                custom_action=CustomAction(
                    action=CustomActionTypes.SCROLL,
                    params={"direction": "up" if clicks > 0 else "down", "clicks": clicks},
                ),
                action_type="scroll", raw=raw,
            )

    elif 'pyautogui.moveTo' in raw_output:
        xy = _xy(raw_output)
        if xy:
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.MOVE_TO, params={"x": xy[0], "y": xy[1]}),
                action_type="move_to", raw=raw,
            )

    elif 'pyautogui.dragTo' in raw_output:
        xy = _xy(raw_output)
        if xy:
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.DRAG_TO, params={"x": xy[0], "y": xy[1]}),
                action_type="drag_to", raw=raw,
            )
    return None

def parse_aguvis_mobile_action(raw_output: str) -> AgentOutput | None:
    # ── mobile / AGUVIS tool calls ────────────────────────────────────────
    raw = {"content": raw_output}
    
    if 'mobile.home' in raw_output:
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.PRESS_HOME),
            action_type="press_home", raw=raw,
        )

    elif 'mobile.back' in raw_output:
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.PRESS_BACK),
            action_type="press_back", raw=raw,
        )

    elif 'mobile.long_press' in raw_output:
        xy = _xy(raw_output)
        if xy:
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.LONG_PRESS, params={"x": xy[0], "y": xy[1]}),
                action_type="long_press", raw=raw,
            )

    elif 'mobile.open_app' in raw_output:
        m = re.search(r"app_name=['\"](.+?)['\"]", raw_output)
        if m:
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.OPEN_APP, params={"app_name": m.group(1)}),
                action_type="open_app", raw=raw,
            )

    elif 'terminate' in raw_output:
        m = re.search(r"status=['\"](.+?)['\"]", raw_output)
        return AgentOutput(
            custom_action=CustomAction(
                action=CustomActionTypes.TERMINATE,
                params={"status": m.group(1) if m else "success"},
            ),
            action_type="terminate", raw=raw,
        )

    elif 'answer' in raw_output:
        m = re.search(r"answer=['\"](.+?)['\"]", raw_output)
        if m:
            return AgentOutput(
                custom_action=CustomAction(action=CustomActionTypes.ANSWER, params={"answer": m.group(1)}),
                action_type="answer", raw=raw,
            )

    return None
