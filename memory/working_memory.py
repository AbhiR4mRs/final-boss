from collections import deque


class WorkingMemory:
    """
    Handles short-term conversational context (Working Memory).
    """
    def __init__(self, max_turns=6):
        self.max_turns = max_turns
        self.history = []

    def add(self, role, content):
        self.history.append({"role": role, "content": content})
        # Keep only the most recent turns
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-(self.max_turns * 2):]

    def context(self):
        """Formats history for the LLM prompt"""
        return "\n".join([f"{m['role']}: {m['content']}" for m in self.history])
 