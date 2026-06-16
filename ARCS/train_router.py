import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import classification_report

# ── 1. Load data ──
train_df = pd.read_csv("router_train.csv")
test_df = pd.read_csv("router_test.csv")

# ── 2. Label mapping ──
LABELS = {"CODING": 0, "MEDICAL": 1, "LEGAL": 2, "GENERAL": 3}
ID2LABEL = {v: k for k, v in LABELS.items()}

train_df["label"] = train_df["label"].map(LABELS)
test_df["label"] = test_df["label"].map(LABELS)

# ── 3. Tokenizer ──
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

# ── 4. Load model ──
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(LABELS),
    id2label=ID2LABEL,
    label2id=LABELS,
)

# ── 5. Metrics ──
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": float(accuracy)}

# ── 6. Training arguments ──
training_args = TrainingArguments(
    output_dir="./router-checkpoints",
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    save_only_model=True,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    logging_steps=10,
    seed=42,
)

# ── 7. Train ──
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# ── 8. Save model (best checkpoint after load_best_model_at_end) ──
trainer.save_model("./router-model")
print("Model saved to ./router-model")

# ── 9. Full evaluation report ──
print("\n── Final Evaluation ──")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print(
    classification_report(
        labels,
        preds,
        target_names=[ID2LABEL[i] for i in range(len(LABELS))],
    )
)
