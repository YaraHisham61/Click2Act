
import json
from typing import Literal
from pydantic import BaseModel, Field
from src.agents.base import AgentOutput, CustomAction, CustomActionTypes


# REFINED [old]: pass → [new]: full implementation mapping ActionSpace + ClickCoordinates to AgentOutput
def parse_holo2_action(raw_output: str) -> AgentOutput | None:
    """Parse a Holo2 NavigationStep, bare ActionSpace, or ClickCoordinates JSON string into AgentOutput."""
    raw = {"content": raw_output}
    try:
        data = json.loads(raw_output)
    except (json.JSONDecodeError, ValueError):
        return None

    # Grounding output: {"x": ..., "y": ...} with no "action" key
    if "x" in data and "y" in data and "action" not in data:
        coords = ClickCoordinates.model_validate(data)
        return AgentOutput(coordinate=(coords.x / 1000, coords.y / 1000), action_type="click", raw=raw)

    # NavigationStep wraps the action; bare action dict also accepted
    action_data = data.get("action", data) if isinstance(data.get("action"), dict) else data
    action_type = action_data.get("action")

    if action_type == "click_element":
        act = ClickElementAction.model_validate(action_data)
        return AgentOutput(coordinate=(act.x / 1000, act.y / 1000), action_type="click", raw=raw)

    elif action_type == "write_element_abs":
        act = WriteElementAction.model_validate(action_data)
        return AgentOutput(coordinate=(act.x / 1000, act.y / 1000), text=act.content, action_type="write", raw=raw)

    elif action_type == "scroll":
        act = ScrollAction.model_validate(action_data)
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.SCROLL, params={"direction": act.direction}),
            action_type="scroll", raw=raw,
        )

    elif action_type == "go_back":
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.GO_BACK),
            action_type="go_back", raw=raw,
        )

    elif action_type == "refresh":
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.REFRESH),
            action_type="refresh", raw=raw,
        )

    elif action_type == "goto":
        act = GotoAction.model_validate(action_data)
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.GOTO, params={"url": act.url}),
            action_type="goto", raw=raw,
        )

    elif action_type == "wait":
        act = WaitAction.model_validate(action_data)
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.WAIT, params={"duration": act.seconds}),
            action_type="wait", raw=raw,
        )

    elif action_type == "restart":
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.RESTART),
            action_type="restart", raw=raw,
        )

    elif action_type == "answer":
        act = AnswerAction.model_validate(action_data)
        return AgentOutput(
            custom_action=CustomAction(action=CustomActionTypes.ANSWER, params={"answer": act.content}),
            action_type="answer", raw=raw,
        )

    return None

# Output for grouning
class ClickCoordinates(BaseModel):
    x: int = Field(ge=0, le=1000, description="The x coordinate, normalized between 0 and 1000.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, normalized between 0 and 1000.")
# Output for navigation
class NavigationStep(BaseModel):
    note: str = Field( default="", description="Task-relevant information extracted from the previous observation. Keep empty if no new info.",)
    thought: str = Field(description="Reasoning about next steps (<4 lines)")
    action: "ActionSpace" = Field(description="Next action to take")
# ========================================================
# Actions
# ========================================================
class ClickElementAction(BaseModel):
    """Click at absolute coordinates of a web element with its description"""

    action: Literal["click_element"] = Field(description="Click at absolute coordinates of a web element")
    element: str = Field(description="text description of the element")
    x: int = Field(ge=0, le=1000, description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, number of pixels from the top edge.")

    def log(self):
        return f"I have clicked on the element '{self.element}' at absolute coordinates {self.x}, {self.y}"


class WriteElementAction(BaseModel):
    """Write content at absolute coordinates of a web element identified by its description, then press Enter."""

    action: Literal["write_element_abs"] = Field(description="Write content at absolute coordinates of a web page")
    content: str = Field(description="Content to write")
    element: str = Field(description="Text description of the element")
    x: int = Field(ge=0, le=1000, description="The x coordinate, number of pixels from the left edge.")
    y: int = Field(ge=0, le=1000, description="The y coordinate, number of pixels from the top edge.")

    def log(self):
        return f"I have written '{self.content}' in the element '{self.element}' at absolute coordinates {self.x}, {self.y}"


class ScrollAction(BaseModel):
    """Scroll action with no required element"""

    action: Literal["scroll"] = Field(
        description="Scroll the page or a specific element"
    )
    direction: Literal["down", "up", "left", "right"] = Field(
        description="The direction to scroll in"
    )

    def log(self):
        return f"I have scrolled {self.direction}"


class GoBackAction(BaseModel):
    """Action to navigate back in browser history"""
    action: Literal["go_back"] = Field(description="Navigate to the previous page")
    def log(self):
        return "I have gone back to the previous page"


class RefreshAction(BaseModel):
    """Action to refresh the current page"""
    action: Literal["refresh"] = Field(description="Refresh the current page")
    def log(self):
        return "I have refreshed the page"


class GotoAction(BaseModel):
    """Action to go to a particular URL"""
    action: Literal["goto"] = Field(description="Goto a particular URL")
    url: str = Field(description="A url starting with http:// or https://")
    def log(self):
        return f"I have navigated to the URL {self.url}"


class WaitAction(BaseModel):
    """Action to wait for a particular amount of time"""
    action: Literal["wait"] = Field(description="Wait for a particular amount of time")
    seconds: int = Field(
        default=2, ge=0, le=10, description="The number of seconds to wait"
    )
    def log(self):
        return f"I have waited for {self.seconds} seconds"

class RestartAction(BaseModel):
    """Restart the task from the beginning."""
    action: Literal["restart"] = "restart"
    def log(self):
        return "I have restarted the task from the beginning"


class AnswerAction(BaseModel):
    """Return a final answer to the task. This is the last action to call in an episode."""
    action: Literal["answer"] = "answer"
    content: str = Field(description="The answer content")
    def log(self):
        return f"I have answered the task with '{self.content}'"


ActionSpace = (
    ClickElementAction
    | WriteElementAction
    | ScrollAction
    | GoBackAction
    | RefreshAction
    | WaitAction
    | RestartAction
    | AnswerAction
    | GotoAction
)
