async def get_summary(model, tokenizer, text: str) -> str:
    inputs = tokenizer.encode("summarize:" + text, return_tensors="pt", max_length=16384, truncation=True)
    summary_ids = model.generate(inputs, max_length=16384, min_length=0, length_penalty=0.5, num_beams=10,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
