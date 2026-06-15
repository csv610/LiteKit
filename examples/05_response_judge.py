"""Example: Evaluate model responses with ResponseJudge."""

from litekit import ResponseJudge, ModelConfig, UserInput

config = ModelConfig(model="gpt-4o-mini", temperature=0.0)
judge = ResponseJudge(model_config=config)

result = judge.evaluate(
    UserInput(
        user_prompt="Explain inheritance in Python.",
        model_response="Inheritance lets a class reuse attributes and methods from another class.",
        ground_truth="Inheritance is an OOP concept where a child class derives properties and behavior from a parent class, enabling code reuse.",
    )
)

print(f"Overall score: {result.overall_score:.2f}")
print(f"Correct: {result.is_correct}")
print(f"Accuracy: {result.criteria.accuracy:.2f}")
print(f"Completeness: {result.criteria.completeness:.2f}")
print(f"Relevance: {result.criteria.relevance:.2f}")
print(f"Clarity: {result.criteria.clarity:.2f}")
print(f"Reasoning: {result.reasoning}")
