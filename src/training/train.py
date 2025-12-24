import yaml
from src.data.load_data import load_resume_dataset
from src.modeling.model import load_model
from src.modeling.tokenizer import load_tokenizer
from src.training.trainer import get_trainer

with open("src/config/model_config.yaml") as f:
    model_cfg = yaml.safe_load(f)

with open("src/config/training_config.yaml") as f:
    train_cfg = yaml.safe_load(f)

dataset = load_resume_dataset(
    "data/processed/train.jsonl",
    "data/processed/validation.jsonl",
    "data/processed/test.jsonl"
)

tokenizer = load_tokenizer(model_cfg["model_name"])

model = load_model(
    model_cfg["model_name"],
    model_cfg["num_labels"],
    model_cfg["id2label"],
    model_cfg["label2id"]
)

trainer = get_trainer(model, tokenizer, dataset, train_cfg)
trainer.train()
trainer.save_model(train_cfg["output_dir"])
