from transformers import AutoModelForSequenceClassification

def load_model(model_name, num_labels, id2label, label2id):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
