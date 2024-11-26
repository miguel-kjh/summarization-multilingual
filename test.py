from litgpt import LLM

llm_l1 = LLM.load("EleutherAI/pythia-160m")
print(llm_l1.model.transformer.wte)

llm_l2 = LLM.load("EleutherAI/pythia-14m")
print(llm_l2.model.transformer.wte)

text = llm_l1.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       