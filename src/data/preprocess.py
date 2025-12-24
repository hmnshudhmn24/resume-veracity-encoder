from src.data.claim_extractor import extract_claims

def preprocess(example):
    example["text"] = extract_claims(example["resume_text"])
    example["label"] = example["label"]
    return example
