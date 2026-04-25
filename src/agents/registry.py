from src.agents.base import GUIAgent
from src.agents.aguvis import AGUVISAgent
from src.agents.halo2 import HALO2Agent
from src.agents.uiagile import UIAgileAgent


MODEL_REGISTRY: dict[str, type[GUIAgent]] = {
    "aguvis": AGUVISAgent,
    "halo2": HALO2Agent,
    "uiagile":UIAgileAgent
}


def build_agent(model_config: dict) -> GUIAgent:
    cls = MODEL_REGISTRY[model_config["class"]]
    agent = cls(model_config.get("params", {}))
    agent.load()
    return agent