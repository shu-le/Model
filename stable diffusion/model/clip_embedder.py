
from typing import List
from torch import nn
from transformers import CLIPTokenizer, CLIPTextModel

class CLIPTextEmbedder(nn.Module):
    
    def __init__(self, version: str = "openai/clip-vit-large-patch14", device="cuda:0", max_length: int = 77):
        
        super().__init__()
        
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        
        self.transformer = CLIPTextModel.from_pretrained(version).eval()
        
        self.device = device
        self.max_length = max_length
        
    def forward(self, prompts: List[str]):
        
        batch_encoding = self.tokenizer(prompts, truncation=True, max_length=self.max_length, return_length=True, 
                                        return_overflowing_token=False, padding="max_length", return_tensors="pt")
        
        tokens = batch_encoding["input_ids"].to(self.device)
        
        return self.transformer(input_ids=tokens).last_hidden_state