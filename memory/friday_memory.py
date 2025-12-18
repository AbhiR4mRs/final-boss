import sqlite3
import os
from datetime import datetime


class FridayMemory:
    """
    FRIDAY Long-Term Memory System
    --------------------------------
    - Stores important user information
    - Uses AI-inspired importance scoring
    - Enforces OWNER-only access
    """

    def __init__(self, db_path="data/friday_memory.db"):
        # Ensure database directory exists safely
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self.db_path = db_path
        self._init_db()

    # -------------------------------
    # Database Initialization
    # -------------------------------
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    memory_type TEXT,
                    importance REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    # -------------------------------
    # AI-Inspired Importance Scoring
    # -------------------------------
    def compute_importance(self, text: str) -> float:
        """
        Estimates importance using linguistic priority cues.
        This simulates selective human-like memory retention.
        """
        keywords = [
            "deadline", "meeting", "exam", "task",
            "important", "urgent", "remember", "must"
        ]

        score = 0.5  # Base importance
        text_lower = text.lower()

        for word in keywords:
            if word in text_lower:
                score += 0.1

        # Cap score to [0, 1]
        return min(score, 1.0)

    # -------------------------------
    # Memory Type Inference
    # -------------------------------
    def infer_memory_type(self, text: str) -> str:
        """
        Categorizes memory into task / preference / fact
        """
        text = text.lower()

        if any(k in text for k in ["deadline", "exam", "meeting", "task"]):
            return "task"
        if any(k in text for k in ["like", "prefer", "favorite"]):
            return "preference"
        return "fact"

    # -------------------------------
    # Store Memory (OWNER ONLY)
    # -------------------------------
    def store_memory(self, content: str, access_mode="GUEST") -> str:
        """
        Stores memory only if:
        - User is OWNER
        - Importance >= threshold
        """
        if access_mode != "OWNER":
            return "Access Denied: I can only store memories for my recognized owner."

        importance = self.compute_importance(content)

        # Threshold chosen empirically to filter low-priority data
        if importance < 0.6:
            return "Okay, noted. I may not prioritize this information."

        memory_type = self.infer_memory_type(content)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO memory (content, memory_type, importance)
                VALUES (?, ?, ?)
                """,
                (content, memory_type, importance)
            )
            conn.commit()

        return "Got it. I've added that to my long-term memory."

    # -------------------------------
    # Recall Memory (OWNER ONLY)
    # -------------------------------
    def recall_memory(self, access_mode="GUEST", limit=5) -> str:
        """
        Retrieves recent stored memories for OWNER only.
        """
        if access_mode != "OWNER":
            return "Privacy Shield: I cannot reveal the owner's personal information."

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT content, memory_type, created_at
                FROM memory
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = cursor.fetchall()

        if not rows:
            return "My memory is currently a blank slate."

        response = "Here is what I remember:\n"
        for idx, (content, mem_type, created) in enumerate(rows, 1):
            response += f"{idx}. ({mem_type}) {content}\n"

        return response


# ----------------------------------
# DEMONSTRATION / TEST BLOCK
# ----------------------------------
if __name__ == "__main__":
    memory = FridayMemory()

    print("=== Scenario 1: Guest tries to store memory ===")
    print(memory.store_memory(
        "My password is admin123",
        access_mode="GUEST"
    ))

    print("\n=== Scenario 2: Owner stores important memory ===")
    print(memory.store_memory(
        "Remember my project presentation is on Monday",
        access_mode="OWNER"
    ))

    print("\n=== Scenario 3: Owner recalls memory ===")
    print(memory.recall_memory(access_mode="OWNER"))

    print("\n=== Scenario 4: Guest tries to recall memory ===")
    print(memory.recall_memory(access_mode="GUEST"))
