from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model_dir = "./models/model"
model_name = "google/flan-t5-small"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
