from transformers import AutoTokenizer, AutoModelForSequenceClassification

def build_model(model_name, num_labels):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    return model, tokenizer
