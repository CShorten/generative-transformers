from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from pydantic import BaseModel

class GenerateInput(BaseModel):
    prompt: str

class Generate:
    model: AutoModelForSeq2SeqLM
    tokenizer: AutoTokenizer
    cuda: bool
    cuda_core: str

    def __init__(self, model_path: str, cuda_support: bool, cuda_core: str):
        self.cuda = cuda_support
        self.cuda_core = cuda_core
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def do(self, item: GenerateInput):
        inputs = self.tokenize(item.prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        print(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))