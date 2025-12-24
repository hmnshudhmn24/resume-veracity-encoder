from transformers import Trainer, TrainingArguments

def get_trainer(model, tokenizer, dataset, cfg):
    args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["batch_size"],
        num_train_epochs=cfg["epochs"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        evaluation_strategy=cfg["evaluation_strategy"],
        save_strategy="epoch"
    )

    return Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer
    )
