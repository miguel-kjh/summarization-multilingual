from evaluate import load
bertscore = load("bertscore")
predictions = ["hello world", "general kenobi"]
references = ["hello land", "Kenobi is a general"]
results = bertscore.compute(
    predictions=predictions, 
    references=references, 
    model_type="distilbert-base-uncased"
)
print(results)