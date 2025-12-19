#This script applies lora fine tunning to for a binary classification problem
#LoRa (low Rank Adaptation)
# The program trains a LLM to performa a classification task on tweets, it should classify tweets in position or negative 
# (1  or 0) respectively.
#Parameter-Efficient Fine-Tuning Methods

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sklearn.metrics import accuracy_score, f1_score
import os
import torch
class LoRaModel:
    def __init__(self):
        # Load the dataset
        self.NUM_TRAIN_EXAMPLES = 1000
        self.NUM_VALIDATION_EXAMPLES = 200
        self.NUM_TEST_EXAMPLES = 200
        self.dataset = None
        self.model_save_path = "./lora_finetuned_model"
        self.model_loaded = False
        self.encoded_dataset = None
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        # Tokenize subsets
        self.model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
        self.lora_config =  LoraConfig( r=8, #rank of matrices typically 4, 8, 16
                                        lora_alpha=16, #Scaling factor of Lora Weights, Controls influence of lora weighs, normally r x 2
                                        target_modules=["q_lin", "v_lin"], # Adjust based on model architecture, apply to attention layers (query, value)
                                        lora_dropout=0.1,# Dropout to prevent overfitting
                                        bias="none", #Whether to train bias terms, none is ost common
                                        task_type=TaskType.SEQ_CLS)
        
    def train_model(self):
        # This Dataset contains thousands of tweets which are classified as positive or negative
        self.dataset = load_dataset("sg247/binary-classification")
        train_subset = self.dataset['train'].select(range(self.NUM_TRAIN_EXAMPLES))
        validation_subset = self.dataset['test'].select(range(self.NUM_VALIDATION_EXAMPLES))
        # Tokenize subsets
        self.encoded_dataset = {
            "train": train_subset.map(self.tokenize, batched=True, remove_columns=train_subset.column_names, load_from_cache_file=False),
            "validation": validation_subset.map(self.tokenize, batched=True, remove_columns=validation_subset.column_names, load_from_cache_file=False)
        }        
        
        # Wrap model with LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        
        # Train the model
        training_args = TrainingArguments(
            output_dir="./sg247",
            eval_strategy="epoch",
            learning_rate=2e-4,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=16,
            num_train_epochs=2,
            weight_decay=0.01, # Regularization Technique, adds a penalty to loss function to keep small weights
            logging_dir="./logs",
            save_strategy="epoch")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.encoded_dataset["train"],
            eval_dataset=self.encoded_dataset["validation"], 
            compute_metrics=self.compute_metrics)
        trainer.train()
        
        # Save the fine-tuned model
        self.model.save_pretrained("./lora_finetuned_model")

        
    # Tokenize
    def tokenize(self, batch):
        texts = [str(t) for t in batch["tweet"]]

        # Replace None labels with a default (e.g., 0) or filter them
        labels = [int(l) if l is not None else 0 for l in batch["label"]]

        tokenized = self.tokenizer( texts,
                                    truncation=True,
                                    padding="max_length",
                                    max_length=512)
        tokenized["labels"] = labels
        return tokenized

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {
            "accuracy": accuracy_score(labels, preds),
            "roc_auc": f1_score(labels, preds)}
        
    def load_model(self):
        if os.path.exists(self.model_save_path):
            print("Loading fine tunned model")
            base_model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
            self.model = PeftModel.from_pretrained(base_model, self.model_save_path)
            self.model_loaded = True
            self.model.eval()
        else:
            print("Model does not exist, select train option first")
            
    def predict_tweet(self, tweet):
        if not self.model_loaded: 
            self.load_model()
        inputs = self.tokenizer(tweet, return_tensors= "pt", truncation=True, padding= "max_length", max_length=512)
    
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class = logits.argmax(dim=-1).item()

        label_map = {0: "negative", 1: "positive"}
        print("Predicted label:", label_map[predicted_class])
        
    def test_batch(self):
        self.dataset = load_dataset("sg247/binary-classification")
        test_set = self.dataset['test'].select(range(self.NUM_TEST_EXAMPLES, 2 * self.NUM_TEST_EXAMPLES))
        labels = test_set['label']
        # Tokenize subsets
        inputs = self.tokenizer(test_set['tweet'], return_tensors= "pt", truncation=True, padding= "max_length", max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        print(accuracy_score(labels, predictions))

if __name__ == "__main__":
    lora_model = LoRaModel()
    while True:
        input_text = input('Select one option:\n1. Test model:\n2. Test batch\n3. Train Model\n(quit for exit):')
        if input_text.lower().strip() == 'quit':
            break
        elif input_text == '1':
            tweet = input("Enter a tweet to classify:")
            lora_model.predict_tweet(tweet)
        elif input_text == '3':
            lora_model.train_model()
        elif input_text == '2':
            lora_model.test_batch()
        else:
            print("Please select a valid option")




