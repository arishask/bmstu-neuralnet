import torch
from transformers import AutoTokenizer, BertForSequenceClassification

class BertClassificationPredictor:
    def __init__(self, pretrained_model="cointegrated/rubert-tiny2-cedr-emotion-detection"):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.model = BertForSequenceClassification.from_pretrained(pretrained_model)

    def predict(self, text):
        tokenized_text = self.tokenizer(text, return_tensors='pt')
        with torch.no_grad():
            pred = self.model(input_ids=tokenized_text.input_ids,
                              attention_mask=tokenized_text.attention_mask,
                              token_type_ids=tokenized_text.token_type_ids)

            prediction_labels = ['no_emotion', 'joy', 'sadness', 'surprise', 'fear', 'anger']
            prediction_values = torch.softmax(pred.logits, -1).cpu().numpy()[0]
            prediction_sorted = sorted([x for x in zip(prediction_labels, prediction_values)],
                                       key=lambda x: x[1], reverse=True)
            prediction_output = "  |  ".join([f"{k}:".ljust(12, ' ') + f"{v:.5f}"
                                              for k, v in prediction_sorted[:3]])

        return prediction_output
