from transformers import pipeline

def load_pipeline(model_path):
    return pipeline(
        "text-classification",
        model=model_path,
        return_all_scores=True
    )
