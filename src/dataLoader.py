"""
Banking77 Arabic Intent Classification Dataset

Loads MSA and dialect variants (Palestinian, Moroccan, Saudi, Tunisian).
Handles bilingual English/Arabic intent labels automatically.
"""

import pandas as pd
from pathlib import Path
from collections import Counter


class Banking77Loader:
    
    def __init__(self, data_dir="datasets"):
        self.data_dir = Path(data_dir)
        self.intent_to_id = {}
        self.id_to_intent = {}
        self.ar_to_en = {}  # Arabic → English lookup
        self.en_to_ar = {}  # English → Arabic lookup
        
        self._load_intents()
    
    def _load_intents(self):
        intent_file = self.data_dir / "Banking77_intents.csv"
        if not intent_file.exists():
            raise FileNotFoundError(f"Missing intent labels: {intent_file}")
        
        labels = pd.read_csv(intent_file)
        unique_intents = sorted(labels['label_en'].unique())
        
        self.intent_to_id = {intent: i for i, intent in enumerate(unique_intents)}
        self.id_to_intent = {i: intent for intent, i in self.intent_to_id.items()}
        
        for _, row in labels.iterrows():
            en, ar = row['label_en'], row['label_ar']
            self.ar_to_en[ar] = en
            self.en_to_ar[en] = ar
        
        print(f"Loaded {len(unique_intents)} intent classes\n")
    
    def _lookup_intent(self, label):
        """Maps English or Arabic label to ID, with fuzzy matching fallback."""
        label = str(label).strip().strip('"\'')
        
        # Try direct English match
        if label in self.intent_to_id:
            return self.intent_to_id[label]
        
        # Try Arabic → English
        if label in self.ar_to_en:
            return self.intent_to_id.get(self.ar_to_en[label])
        
        # Fuzzy match as last resort (handles partial labels)
        for intent in self.intent_to_id:
            if intent in label or label in intent:
                return self.intent_to_id[intent]
        
        for ar_label, en_label in self.ar_to_en.items():
            if ar_label in label or label in ar_label:
                return self.intent_to_id.get(en_label)
        
        return None
    
    def _find_columns(self, df):
        cols = df.columns.tolist()
        
        # Find intent column
        label_col = next(
            (c for c in cols if any(k in c.lower() for k in ['intent', 'label', 'category'])),
            cols[0]
        )
        
        # Find text columns (skip IDs and labels)
        text_cols = [
            c for c in cols 
            if any(k in c.lower() for k in ['question', 'query', 'text', 'utterance'])
            and 'id' not in c.lower()
        ]
        
        if not text_cols:
            text_cols = [c for c in cols if c != label_col and 'id' not in c.lower()]
        
        return label_col, text_cols
    
    def _parse_split(self, df, dialect="all"):
        label_col, text_cols = self._find_columns(df)
        
        # Filter by dialect if needed
        if dialect == "msa":
            text_cols = [c for c in text_cols if 'msa' in c.lower()]
        elif dialect in ["pal", "palestinian"]:
            text_cols = [c for c in text_cols if 'pal' in c.lower()]
        
        if not text_cols:
            raise ValueError(f"No text columns for dialect: {dialect}")
        
        texts, labels = [], []
        skipped = set()
        
        for _, row in df.iterrows():
            if pd.isna(row[label_col]):
                continue
            
            intent_id = self._lookup_intent(row[label_col])
            if intent_id is None:
                skipped.add(str(row[label_col]).strip())
                continue
            
            for col in text_cols:
                if col in row and pd.notna(row[col]):
                    text = str(row[col]).strip()
                    if text:
                        texts.append(text)
                        labels.append(intent_id)
        
        if skipped:
            print(f"  Warning: Skipped {len(skipped)} unknown intents")
        
        return texts, labels
    
    def load_train(self, dialect="all"):
        path = self.data_dir / "Banking77_Arabized_MSA_PAL_train.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        
        df = pd.read_csv(path, on_bad_lines='skip')
        texts, labels = self._parse_split(df, dialect)
        
        top_labels = dict(Counter(labels).most_common(3))
        print(f"Train: {len(texts):,} samples | Top: {top_labels}")
        return texts, labels
    
    def load_val(self, dialect="all"):
        path = self.data_dir / "Banking77_Arabized_MSA_PAL_val.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        
        df = pd.read_csv(path, on_bad_lines='skip')
        texts, labels = self._parse_split(df, dialect)
        
        print(f"Val: {len(texts):,} samples | Top: {dict(Counter(labels).most_common(3))}")
        return texts, labels
    
    def load_test(self, dialect="msa"):
        dialect_map = {
            "msa": "Banking77_Arabized_MSA_test.csv",
            "pal": "Banking77_Arabized_PAL_test.csv",
            "palestinian": "Banking77_Arabized_PAL_test.csv",
            "moroccan": "Banking77_Arabized_Moroccan_test.csv",
            "saudi": "Banking77_Arabized_Saudi_test.csv",
            "tunisian": "Banking77_Arabized_Tunisian_test.csv",
        }
        
        filename = dialect_map.get(dialect.lower())
        if not filename:
            raise ValueError(f"Unknown dialect: {dialect}. Options: {list(dialect_map.keys())}")
        
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        
        df = pd.read_csv(path, on_bad_lines='skip')
        texts, labels = self._parse_split(df, "all")
        
        print(f"Test ({dialect}): {len(texts):,} samples | Top: {dict(Counter(labels).most_common(3))}")
        return texts, labels
    
    def load_all(self):
        print("=" * 50)
        print("Loading Banking77 Dataset")
        print("=" * 50 + "\n")
        
        data = {}
        
        # Load train/val
        for split in ['train', 'val']:
            try:
                loader = getattr(self, f'load_{split}')
                texts, labels = loader("all")
                data[split] = {'texts': texts, 'labels': labels}
            except Exception as e:
                print(f"Skipping {split}: {e}\n")
        
        # Load all test dialects
        for dialect in ["msa", "pal", "moroccan", "saudi", "tunisian"]:
            try:
                texts, labels = self.load_test(dialect)
                data[f'test_{dialect}'] = {'texts': texts, 'labels': labels}
            except Exception as e:
                print(f"Skipping {dialect} test: {e}\n")
        
        self._print_summary(data)
        return data
    
    def _print_summary(self, data):
        print("\n" + "=" * 50)
        print("Dataset Summary")
        print("=" * 50)
        
        total = sum(len(split['texts']) for split in data.values())
        
        for name, split in data.items():
            print(f"  {name:20s} {len(split['texts']):>6,} samples")
        
        print(f"\n  {'Total':20s} {total:>6,} samples")
        print(f"  {'Classes':20s} {len(self.intent_to_id):>6,}")
        print("=" * 50 + "\n")
    
    def get_intent_name(self, intent_id, lang="en"):
        """Returns intent label in English or Arabic."""
        en_name = self.id_to_intent.get(intent_id, "unknown")
        return self.en_to_ar.get(en_name, en_name) if lang == "ar" else en_name
    
    def get_intent_id(self, label):
        """Returns numeric ID for English or Arabic intent label."""
        return self._lookup_intent(label) or -1


if __name__ == "__main__":
    loader = Banking77Loader("datasets")
    
    # Test bilingual lookup
    print("Testing bilingual resolution:")
    print(f"  'card_arrival' → ID {loader.get_intent_id('card_arrival')}")
    print(f"  'وصول البطاقة' → ID {loader.get_intent_id('وصول البطاقة')}\n")
    
    dataset = loader.load_all()
    
    if 'train' in dataset:
        print("Sample training examples:")
        for i in range(min(3, len(dataset['train']['texts']))):
            text = dataset['train']['texts'][i]
            label_id = dataset['train']['labels'][i]
            en = loader.get_intent_name(label_id)
            ar = loader.get_intent_name(label_id, lang="ar")
            print(f"  [{label_id}] {en} / {ar}")
            print(f"      {text[:60]}...")