"""Lightweight conversation memory: keeps recent Q&A turns for context."""


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self.turns: list[dict] = []

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append({"question": question, "answer": answer})
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def get_history_text(self) -> str:
        if not self.turns:
            return "No previous conversation."
        lines = []
        for t in self.turns:
            lines.append(f"User: {t['question']}")
            lines.append(f"Assistant: {t['answer']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.turns = []