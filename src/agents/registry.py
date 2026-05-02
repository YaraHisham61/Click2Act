from src.agents.base import GUIAgent
from src.agents.aguvis import AGUVISAgent
from src.agents.holo2 import HOLO2Agent
from src.agents.uiagile import UIAgileAgent


MODEL_REGISTRY: dict[str, type[GUIAgent]] = {
    "aguvis": AGUVISAgent,
    "holo2": HOLO2Agent,
    "uiagile":UIAgileAgent
}


def build_agent(model_config: dict) -> GUIAgent:
    cls = MODEL_REGISTRY[model_config["class"]]
    agent = cls(model_config.get("params", {}))
    agent.load()
    return agent