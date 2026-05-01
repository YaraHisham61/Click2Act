from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from PIL.Image import Image
from enum import Enum

class CustomActionTypes(str, Enum):
    # ── Mobile ───────────────────────────────────────────────────────────────
    PRESS_HOME  = "press_home"
    PRESS_BACK  = "press_back"
    PRESS_ENTER = "press_enter"
    LONG_PRESS  = "long_press"   # params: x, y (normalized)
    SCROLL      = "scroll"       # params: direction ("up"|"down"|"left"|"right"), optional x, y
    SWIPE       = "swipe"        # params: start_x, start_y, end_x, end_y (normalized)
    OPEN_APP    = "open_app"     # params: app_name
    TERMINATE   = "terminate"    # params: status ("success"|"failure"|"impossible")
    ANSWER      = "answer"       # params: answer  — QA / AGUVIS answer tool
    WAIT        = "wait"         # params: duration (seconds, optional)
    # ── Desktop / PyAutoGUI ──────────────────────────────────────────────────
    MOVE_TO     = "move_to"      # params: x, y
    KEY_PRESS   = "key_press"    # params: key  e.g. "esc", "tab", "space"
    HOTKEY      = "hotkey"       # params: keys (list)  e.g. ["ctrl", "c"]
    DRAG_TO     = "drag_to"      # params: x, y, optional button, duration


class CustomAction(BaseModel):
    action: CustomActionTypes
    # Keys vary per action; see CustomActionTypes inline comments
    params: dict | None = Field(default=None)


class AgentOutput(BaseModel):
    coordinate: tuple[float, float] | None = Field(default=None, description="(x,y) normalized, for click or grounding")
    action_type: str | None = Field(default=None, description="action type to parse it - later for MMM-Benchmarks or OS-World")
    text: str | None = Field(default=None, description="if there's text need to written")
    custom_action: CustomAction | None = Field(default=None, description="non-tap mobile/OS action (home, back, scroll, terminate, …)")
    raw: dict | None = Field(default=None, description="raw output from agent - for debugging")

class GUIAgent(ABC):
    def __init__(self, config: dict[str, any]) -> None:
        self.config = config
        
    @abstractmethod
    def load(self) -> None:
        """Load agent"""
    
    def predict_click(self, screenshot: Image, task: str) -> AgentOutput:
        """predict click normalized (x,y) - for benchmarks like ScreenSpot"""
        raise NotImplementedError()

    def predict_click_batch(self, inputs: list[tuple[Image, str]]) -> list[AgentOutput]:
        """Batch version of predict_click. Override in subclasses for true batch inference."""
        return [self.predict_click(screenshot, task) for screenshot, task in inputs]
    
    def predict_action(self, screenshot: Image, task: str) -> AgentOutput:
        """predict action - for benchmarks like MMBenchGUI"""
        raise NotImplementedError()
    
    def predict_stateful(self, screenshot: Image, task: str, history: list[AgentOutput]) -> AgentOutput:
        """Multi-step actions with history - for benchmarks like OSWorld"""
        raise NotImplementedError()
    
    # Template method hooks — override in subclasses when needed                                                                                                                
    def preprocess(self, screenshot: Image) -> Image:                                                                                                       
        return screenshot                                                                                                                                                       
                                                                                                                                                                                
    def postprocess(self, raw_output: any) -> AgentOutput:                                                                                                                      
        raise NotImplementedError                  