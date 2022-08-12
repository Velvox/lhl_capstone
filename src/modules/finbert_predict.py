from finbert.finbert import predict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

model_fp = "/Users/niclapenna/Documents/GitHub/finBERT/models/classifier_model/finbert-sentiment"
text_fp = '/Users/niclapenna/Documents/GitHub/lhl_capstone/data/news/to_classify/description_processed.txt'
output_fp = '/Users/niclapenna/Documents/GitHub/lhl_capstone/data/news/classified'


with open(text_fp,'r') as f:
    text = f.read()

model = AutoModelForSequenceClassification.from_pretrained(model_fp,num_labels=3,cache_dir=None)


output = "predictions.csv"
predict(text,model,write_to_csv=True,path=os.path.join(output_fp,output))