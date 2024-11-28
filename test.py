"""from litgpt import LLM

llm_l1 = LLM.load("EleutherAI/pythia-160m")
print(llm_l1.model.transformer.wte)

llm_l2 = LLM.load("EleutherAI/pythia-14m")
print(llm_l2.model.transformer.wte)

text = llm_l1.generate("Fix the spelling: Every fall, the familly goes to the mountains.")
print(text)
# Corrected Sentence: Every fall, the family goes to the mountains.       """

import torch
import litgpt
from litgpt import LLM
from litgpt.lora import mark_only_lora_as_trainable
from litgpt.data import Alpaca2k
import lightning as L

batch_size = 8
accumulate_grad_batches = 1

class LitLLM(L.LightningModule):
    def __init__(self, checkpoint_dir, tokenizer_dir=None, trainer_ckpt_path=None):
        super().__init__()
 
        self.llm = LLM.load(checkpoint_dir, tokenizer_dir=tokenizer_dir, distribute=None)
        self.trainer_ckpt_path = trainer_ckpt_path

    def setup(self, stage):
        self.llm.trainer_setup(trainer_ckpt=self.trainer_ckpt_path)
        
    def training_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch):
        logits, loss = self.llm(input_ids=batch["input_ids"], target_ids=batch["labels"])
        self.log("validation_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        warmup_steps = 10
        optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=0.0002, weight_decay=0.0, betas=(0.9, 0.95))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: step / warmup_steps)
        return [optimizer], [scheduler]
    
# main
if __name__ == "__main__":
    
    lit_model = LitLLM(checkpoint_dir="EleutherAI/pythia-160m")
    data = Alpaca2k()

    data.connect(lit_model.llm.tokenizer, batch_size=batch_size, max_seq_length=512)

    trainer = L.Trainer(
        devices=1,
        accelerator="cuda",
        max_epochs=1,
        accumulate_grad_batches=accumulate_grad_batches,
        precision="bf16-true",
    )
    trainer.fit(lit_model, data)

    lit_model.llm.model.to(lit_model.llm.preprocessor.device)
    lit_model.llm.generate("hello world")