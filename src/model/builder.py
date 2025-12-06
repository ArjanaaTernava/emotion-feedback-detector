from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Build a Transformer-based classification model and its tokenizer
def build_model(model_name, num_labels):
    # Load the tokenizer for the given pretrained model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load the pretrained model and adjust it for the desired number of labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Return both model and tokenizer
    return model, tokenizer
