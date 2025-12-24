from src.inference.pipeline import load_pipeline

def predict(resume_text, model_path="checkpoints"):
    pipe = load_pipeline(model_path)
    return pipe(resume_text)
