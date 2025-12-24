def extract_claims(resume_text):
    lines = resume_text.split("\n")
    claims = []
    for line in lines:
        if any(k in line.lower() for k in ["experience", "worked", "developed", "managed", "led"]):
            claims.append(line.strip())
    return " ".join(claims)
