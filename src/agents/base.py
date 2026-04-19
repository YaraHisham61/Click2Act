from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from PIL.Image import Image


class AgentOutput(BaseModel):
    coordinate: tuple[float, float] | None = Field(default=None, description="(x,y) normalized, for click or grounding")
    action_type: str | None = Field(default=None, description="action type to parse it - later for MMM-Benchmarks or OS-World")
    text: str | None = Field(default=None, description="if there's text need to written")
    raw: dict | None = Field(default=None, description="raw output from agent - for debugging")

class GUIAgent(ABC):
    def __init__(self, config: dict[str, any]) -> None:
        self.config = config
        
    @abstractmethod
    def load(self) -> None:
        """Load agent"""
    
    def predict_click(sel, screenshot: Image, task: str) -> AgentOutput:
        """predict click normalized (x,y) - for benchmarks like ScreenSpot"""
        raise NotImplementedError()
    
    def predict_action(sel, screenshot: Image, task: str) -> AgentOutput:
        """predict action - for benchmarks like MMBenchGUI"""
        raise NotImplementedError()
    
    def predict_stateful(sel, screenshot: Image, task: str, history: list[AgentOutput]) -> AgentOutput:
        """Multi-step actions with history - for benchmarks like OSWorld"""
        raise NotImplementedError()
    
    # Template method hooks — override in subclasses when needed                                                                                                                
    def preprocess(self, screenshot: Image) -> Image:                                                                                                       
        return screenshot                                                                                                                                                       
                                                                                                                                                                                
    def postprocess(self, raw_output: any) -> AgentOutput:                                                                                                                      
        raise NotImplementedError                  