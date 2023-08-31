from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BertTokenizer, 
    BertForSequenceClassification,
    GPT2Tokenizer,
    GPT2ForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)

def model_RoBERTa():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    return tokenizer, model

def model_textattack():
    tokenizer = AutoTokenizer.from_pretrained("textattack/roberta-base-SST-2")
    model = AutoModelForSequenceClassification.from_pretrained("textattack/roberta-base-SST-2")
    return tokenizer, model

def model_BERT():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    return tokenizer, model

def model_GPT2():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2ForSequenceClassification.from_pretrained("gpt2")
    return tokenizer, model

def model_DistilBERT():
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
    return tokenizer, model

def get_model_and_tokenizer(model_type):
    if model_type == "roberta":
        return model_RoBERTa()
    elif model_type == "textattack":
        return model_textattack()
    elif model_type == "bert":
        return model_BERT()
    elif model_type == "gpt2":
        return model_GPT2()
    elif model_type == "distilbert":
        return model_DistilBERT()
    else:
        raise ValueError("Invalid model type")
