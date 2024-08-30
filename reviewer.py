from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import Trainer, TrainingArguments, BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import sys
import argparse



model_path = "./results"


class JapaneseSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=128)
        item = {key: torch.tensor(val) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def train(): 
    model_name = "cl-tohoku/bert-base-japanese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)
    
    texts = []
    labels = []

    with open('training.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

    for line in lines:
        line = line.strip() 
        if line:  
            label, text = line.split(',', 1) 
            labels.append(int(label))  
            texts.append(text) 
    
    dataset = JapaneseSentimentDataset(texts, labels, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=model_path,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=5e-5,
        warmup_steps=0,
        weight_decay=0.01,
        eval_strategy='no',
    )

    trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    )

    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()

    trainer.save_model("./results")
    tokenizer.save_pretrained("./results")


def rate():
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)
    model.eval()


    sentiment_analyzer = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


    with open('reviews.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()

        total = 0
        score = 0
        for line in lines:
            line = line.strip() 
            score += sentiment_analyzer(line)[0]['score']
            total +=1

    final_score = (score/total) * 100
    print(final_score)






def main():
    parser = argparse.ArgumentParser(description="Bookmeter book rater")

    parser.add_argument('--input', type=str, help='Link to bookmeter', required=True)
    parser.add_argument('--train', action='store_true', help='Whether or not to train the mode. Must be done the first use')
    args = parser.parse_args()

    if args.train:
        train()
    parse(args.input)
    rate()




def parse(link):
    driver = webdriver.Chrome() 
    driver.get(link)
    try:    
        div = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'section.bm-page-loader[data-status=idle]')))
        spans = div.find_elements(By.TAG_NAME, 'span')
        with open('reviews.txt', 'w', encoding='utf-8') as txtfile:
            for span in spans:
                text = span.text
                if text != '' and text != 'ネタバレ' and text != 'ナイス' and text.isdigit() == False:
                    txtfile.write(text)
                    txtfile.write("\n")
    finally:
        driver.quit()




if __name__=="__main__":
    main()