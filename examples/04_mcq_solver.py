"""Example: Solve multiple-choice questions with LiteMCQClient."""

from litekit import LiteMCQClient, MCQInput

solver = LiteMCQClient(model="gpt-4o-mini", temperature=0.1)

question = MCQInput(
    question="What is the primary purpose of a cache in a computer system?",
    options={
        "A": "To increase the amount of RAM",
        "B": "To speed up data access by storing frequently used data",
        "C": "To replace the CPU",
        "D": "To store the operating system permanently",
    },
    context="Computer Architecture",
)

answer = solver.solve(question)
if answer:
    print(f"Question: {answer.question}")
    print(f"Correct: {[o.key for o in answer.correct_options]}")
    print(f"Reasoning: {answer.reasoning}")
    print(f"Confidence: {answer.confidence:.2f}")
else:
    print("Failed to solve the question.")
