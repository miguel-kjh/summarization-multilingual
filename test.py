from litgpt import LLM

llm = LLM.load("EleutherAI/pythia-14m")
text = llm.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       