"""Step 7: an LLM grades each answer for correctness and faithfulness (the guide's judge, on Ollama).

A model grading answers can be lenient, especially when it's the same model that wrote them;
eval/hand_grade.py measures how often it agrees with you.
"""
import ollama
from pydantic import BaseModel

from rag.generate import Answer

JUDGE_MODEL = "qwen3:14b"     # same model as the generator; qwen3:4b is faster


class Verdict(BaseModel):
    correct: bool     # states the same facts as the expected answer (wording may differ)
    faithful: bool    # every claim in the answer is supported by the cited quotes
    reason: str       # one sentence, so you can read why it failed


JUDGE = """You grade answers from a tax question-answering system.
correct: the system answer states the same facts as the expected answer; wording may differ.
faithful: every claim in the system answer is supported by the cited quotes.
Give a one-sentence reason. Respond in JSON."""


def judge(question: str, expected: str, answer: Answer) -> Verdict:
    quotes = "\n".join(f"- [{c.chunk_id}] {c.quote}" for c in answer.citations) or "(none)"
    response = ollama.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "system", "content": JUDGE},
                  {"role": "user", "content":
                   f"Question: {question}\nExpected answer: {expected}\n"
                   f"System answer: {answer.answer}\nCited quotes:\n{quotes}"}],
        format=Verdict.model_json_schema(),
        options={"temperature": 0},
        think=False,
    )
    return Verdict.model_validate_json(response.message.content)
