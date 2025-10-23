from abc import ABC, abstractmethod
from typing import List, Dict, Any

class Retriever(ABC):
    @abstractmethod
    def build(self, docs: List[Dict[str, Any]]) -> None:
        ...
    @abstractmethod
    def query(self, q: str, topk: int = 5) -> List[Dict[str, Any]]:
        ...
    @property
    @abstractmethod
    def name(self) -> str:
        ...
