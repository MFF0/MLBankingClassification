"""
Banking77 Arabic Intent Classification Models

Three transformer models for ensemble prediction:
- AraBERT: aubmindlab/bert-base-arabertv2
- MARBERT: UBC-NLP/MARBERT  
- AraT5: UBC-NLP/AraT5-base (text-to-text approach)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os


class IntentDataset(Dataset):
    """Standard BERT-style dataset for intent classification."""
    
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class T5IntentDataset(Dataset):
    """T5 dataset with text-to-text formatting."""
    
    def __init__(self, texts, labels, tokenizer, id_to_intent, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.id_to_intent = id_to_intent
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # T5 uses task prefix
        input_text = f"classify intent: {self.texts[idx]}"
        target = self.id_to_intent[self.labels[idx]]
        
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            target,
            max_length=32,  # Intent names are short
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Mask padding in target (loss ignores -100)
        target_ids = targets['input_ids'].flatten()
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'labels': target_ids
        }


class AraBERTClassifier:
    """AraBERT-based intent classifier."""
    
    def __init__(self, num_classes=77, max_len=128, model_name="aubmindlab/bert-base-arabertv2"):
        self.num_classes = num_classes
        self.max_len = max_len
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\nLoading {model_name} on {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        ).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {n_params:,} parameters\n")
    
    def train(self, train_texts, train_labels, val_texts, val_labels,
              batch_size=16, epochs=3, lr=2e-5, save_path="models/arabert.pt"):
        
        print("=" * 60)
        print("Training AraBERT")
        print("=" * 60)
        
        train_data = IntentDataset(train_texts, train_labels, self.tokenizer, self.max_len)
        val_data = IntentDataset(val_texts, val_labels, self.tokenizer, self.max_len)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_loader) * epochs
        )
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            self.model.train()
            train_loss = 0
            correct, total = 0, 0
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids=input_ids, attention_mask=mask, labels=labels)
                
                loss = outputs.loss
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.3f}', 'acc': f'{correct/total:.3f}'})
            
            train_acc = correct / total
            val_acc = self._validate(val_loader)
            
            print(f"Train: loss={train_loss/len(train_loader):.3f}, acc={train_acc:.3f}")
            print(f"Val: acc={val_acc:.3f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved checkpoint (val_acc={val_acc:.3f})")
        
        print(f"\nBest validation accuracy: {best_val_acc:.3f}")
        return best_val_acc
    
    def _validate(self, loader):
        self.model.eval()
        correct, total = 0, 0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=mask)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        return correct / total
    
    def predict(self, texts, batch_size=32):
        self.model.eval()
        dataset = IntentDataset(texts, [0] * len(texts), self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        preds = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        
        return np.array(preds)
    
    def predict_proba(self, texts, batch_size=32):
        self.model.eval()
        dataset = IntentDataset(texts, [0] * len(texts), self.tokenizer, self.max_len)
        loader = DataLoader(dataset, batch_size=batch_size)
        
        all_probs = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing probabilities"):
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids=input_ids, attention_mask=mask).logits
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_probs)
    
    def evaluate(self, texts, labels, batch_size=32):
        preds = self.predict(texts, batch_size)
        acc = accuracy_score(labels, preds)
        print(f"\nTest accuracy: {acc:.3f}")
        return acc
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


class MARBERTClassifier(AraBERTClassifier):
    """MARBERT classifier - identical architecture to AraBERT."""
    
    def __init__(self, num_classes=77, max_len=128):
        super().__init__(num_classes, max_len, model_name="UBC-NLP/MARBERT")


class AraT5Classifier:
    """AraT5 text-to-text intent classifier."""
    
    def __init__(self, num_classes=77, intent_to_id=None, id_to_intent=None, max_len=128):
        self.num_classes = num_classes
        self.max_len = max_len
        self.intent_to_id = intent_to_id or {}
        self.id_to_intent = id_to_intent or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model_name = "UBC-NLP/AraT5-base"
        print(f"\nLoading {model_name} on {self.device}")
        
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model loaded: {n_params:,} parameters\n")
    
    def train(self, train_texts, train_labels, val_texts, val_labels,
              batch_size=8, epochs=3, lr=3e-4, save_path="models/arat5.pt"):
        
        print("=" * 60)
        print("Training AraT5")
        print("=" * 60)
        
        train_data = T5IntentDataset(train_texts, train_labels, self.tokenizer, 
                                      self.id_to_intent, self.max_len)
        val_data = T5IntentDataset(val_texts, val_labels, self.tokenizer, 
                                    self.id_to_intent, self.max_len)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=len(train_loader) * epochs
        )
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            self.model.train()
            train_loss = 0
            
            pbar = tqdm(train_loader, desc="Training")
            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                loss = self.model(input_ids=input_ids, attention_mask=mask, labels=labels).loss
                
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                pbar.set_postfix({'loss': f'{loss.item():.3f}'})
            
            val_acc = self._validate(val_texts, val_labels)
            
            print(f"Train: loss={train_loss/len(train_loader):.3f}")
            print(f"Val: acc={val_acc:.3f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved checkpoint (val_acc={val_acc:.3f})")
        
        print(f"\nBest validation accuracy: {best_val_acc:.3f}")
        return best_val_acc
    
    def _validate(self, texts, labels):
        preds = self.predict(texts)
        return accuracy_score(labels, preds)
    
    def predict(self, texts, batch_size=16):
        self.model.eval()
        preds = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i+batch_size]
            inputs = [f"classify intent: {text}" for text in batch]
            
            tokens = self.tokenizer(
                inputs,
                max_length=self.max_len,
                padding=True,
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **tokens,
                    max_length=32,
                    num_beams=4,
                    early_stopping=True
                )
            
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Map intent names back to IDs
            for intent_name in decoded:
                preds.append(self.intent_to_id.get(intent_name, 0))
        
        return np.array(preds)
    
    def predict_proba(self, texts, batch_size=16):
        # T5 doesn't naturally output probabilities, so return one-hot
        preds = self.predict(texts, batch_size)
        probs = np.zeros((len(preds), self.num_classes))
        probs[np.arange(len(preds)), preds] = 1.0
        return probs
    
    def evaluate(self, texts, labels, batch_size=16):
        preds = self.predict(texts, batch_size)
        acc = accuracy_score(labels, preds)
        print(f"\nTest accuracy: {acc:.3f}")
        return acc
    
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
    
    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


if __name__ == "__main__":
    print("\nBanking77 Transformer Models")
    print("Available models:")
    print("  • AraBERT  - aubmindlab/bert-base-arabertv2")
    print("  • MARBERT  - UBC-NLP/MARBERT")
    print("  • AraT5    - UBC-NLP/AraT5-base")
    print("\nSee documentation for usage examples.")