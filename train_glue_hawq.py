import argparse
import os
import random
import time
import warnings
import subprocess
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup
)

import evaluate
from tqdm import tqdm


warnings.filterwarnings("ignore")


# =========================================================
# ARGUMENTS
# =========================================================

parser = argparse.ArgumentParser(description='Improved HAWQ GLUE Training')

parser.add_argument('--task',
                    type=str,
                    default='sst2',
                    choices=['sst2', 'qnli', 'mnli', 'qqp', 'rte','mrpc','wnli','stsb','cola'])

parser.add_argument('--model',
                    type=str,
                    default='bert-base-uncased')

parser.add_argument('--epochs',
                    type=int,
                    default=1)

parser.add_argument('--batch-size',
                    type=int,
                    default=8)

parser.add_argument('--lr',
                    type=float,
                    default=3e-5)

parser.add_argument('--max-length',
                    type=int,
                    default=256)

parser.add_argument('--weight-bit',
                    type=int,
                    default=8)

parser.add_argument('--activation-bit',
                    type=int,
                    default=8)

parser.add_argument('--seed',
                    type=int,
                    default=42)

parser.add_argument('--save-path',
                    type=str,
                    default='./checkpoints/')

parser.add_argument('--csv-path',
                    type=str,
                    default='./results.csv')

args = parser.parse_args()


# =========================================================
# RANDOM SEED
# =========================================================

random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


# =========================================================
# DEVICE
# =========================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nUsing device: {device}")


# =========================================================
# TASK CONFIG
# =========================================================

task_to_keys = {
    'sst2': ('sentence', None),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'mnli': ('premise', 'hypothesis'),
    'rte': ('sentence1', 'sentence2'),
    'mrpc': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
    'stsb': ('sentence1', 'sentence2'),
    'cola': ('sentence', None)
}

num_labels_dict = {
    'sst2': 2,
    'qnli': 2,
    'qqp': 2,
    'mnli': 3,
    'rte': 2,
    'mrpc': 2,
    'wnli': 2,
    'cola': 2,
    'stsb': 1
}


# =========================================================
# LOAD DATASET
# =========================================================

print(f"Loading GLUE dataset: {args.task}")

dataset = load_dataset("glue", args.task)

sentence1_key, sentence2_key = task_to_keys[args.task]


# =========================================================
# TOKENIZER
# =========================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(args.model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):

    if sentence2_key is None:

        return tokenizer(
            examples[sentence1_key],
            truncation=True,
            max_length=args.max_length
        )

    return tokenizer(
        examples[sentence1_key],
        examples[sentence2_key],
        truncation=True,
        max_length=args.max_length
    )


print("Tokenizing dataset...")

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)


# =========================================================
# DATASET FORMAT
# =========================================================

train_dataset = tokenized_dataset['train']

if args.task == 'mnli':
    eval_dataset = tokenized_dataset['validation_matched']
else:
    eval_dataset = tokenized_dataset['validation']


train_dataset = train_dataset.rename_column(
    "label",
    "labels"
)

eval_dataset = eval_dataset.rename_column(
    "label",
    "labels"
)

train_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)

eval_dataset.set_format(
    type='torch',
    columns=['input_ids', 'attention_mask', 'labels']
)


# =========================================================
# DATALOADER
# =========================================================

collator = DataCollatorWithPadding(tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collator
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    collate_fn=collator
)


# =========================================================
# MODEL
# =========================================================

print(f"Loading model: {args.model}")

if args.task == "stsb":

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=1,
        problem_type="regression"
    )

else:

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=num_labels_dict[args.task]
    )

if args.model == "gpt2":
    model.config.pad_token_id = tokenizer.eos_token_id


# =========================================================
# ACTIVATION QUANTIZATION
# =========================================================

class QuantAct(nn.Module):

    def __init__(self, activation_bit=8):

        super().__init__()

        self.activation_bit = activation_bit

    def forward(self, x):

        qmin = -(2 ** (self.activation_bit - 1))
        qmax = (2 ** (self.activation_bit - 1)) - 1

        scale = x.abs().max() / qmax

        if scale == 0:
            return x

        x_int = torch.clamp(
            (x / scale).round(),
            qmin,
            qmax
        )

        return x_int * scale


# =========================================================
# QUANT LINEAR
# =========================================================

class QuantLinear(nn.Module):

    def __init__(self,
                 layer,
                 weight_bit=8,
                 activation_bit=8):

        super().__init__()

        self.layer = layer

        self.weight_bit = weight_bit

        self.activation_quant = QuantAct(
            activation_bit
        )

    def quantize_weight(self, x):

        qmin = -(2 ** (self.weight_bit - 1))
        qmax = (2 ** (self.weight_bit - 1)) - 1

        scale = x.abs().max() / qmax

        if scale == 0:
            return x

        x_int = torch.clamp(
            (x / scale).round(),
            qmin,
            qmax
        )

        return x_int * scale

    def forward(self, x):

#       x = self.activation_quant(x)

        quant_weight = self.quantize_weight(
            self.layer.weight
        )

        return nn.functional.linear(
            x,
            quant_weight,
            self.layer.bias
        )


# =========================================================
# HAWQ STYLE MIXED PRECISION
# =========================================================

def quantize_model(module):

    for name, child in module.named_children():

        if isinstance(child, nn.Linear):

            if "classifier" in name:
                continue

            bit = args.weight_bit

            if "attention" in name:
                bit = 8

            elif "intermediate" in name:
                bit = 4

            setattr(
                module,
                name,
                QuantLinear(
                    child,
                    weight_bit=bit,
                    activation_bit=args.activation_bit
                )
            )

        else:
            quantize_model(child)


print("Applying quantization...")

quantize_model(model)


# =========================================================
# MODEL TO DEVICE
# =========================================================

model = model.to(device)


# =========================================================
# MODEL MEMORY SIZE
# =========================================================

def get_model_size(model):

    param_size = 0

    for param in model.parameters():

        param_size += (
            param.nelement() *
            param.element_size()
        )

    buffer_size = 0

    for buffer in model.buffers():

        buffer_size += (
            buffer.nelement() *
            buffer.element_size()
        )

    size_mb = (
        param_size + buffer_size
    ) / 1024**2

    return size_mb


model_size = get_model_size(model)

print(f"\nModel Size: {model_size:.2f} MB")

# =========================================================
# GPU POWER USAGE
# =========================================================

def get_gpu_power():

    try:

        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=power.draw",
                "--format=csv,noheader,nounits"
            ]
        )

        power = result.decode("utf-8").strip()

        return float(power)

    except:

        return -1
# =========================================================
# ENERGY CONSUMPTION
# =========================================================

energy_consumption = get_gpu_power()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr
)

total_steps = len(train_loader) * args.epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

# =========================================================
# AMP
# =========================================================

scaler = GradScaler()


# =========================================================
# TRAIN
# =========================================================

def train(epoch):

    model.train()

    total_loss = 0

    progress_bar = tqdm(train_loader)

    for batch in progress_bar:

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        optimizer.zero_grad()

        with autocast():

            outputs = model(**batch)

            loss = outputs.loss

        scaler.scale(loss).backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        total_loss += loss.item()

        progress_bar.set_description(
            f"Epoch {epoch} Loss {loss.item():.4f}"
        )

    avg_loss = total_loss / len(train_loader)

    print(f"\nTrain Loss: {avg_loss:.4f}")

    return avg_loss


# =========================================================
# VALIDATION
# =========================================================

def validate():

    model.eval()

    metric = evaluate.load("glue", args.task)

    total_latency = 0

    total_samples = 0

    all_predictions = []

    all_labels = []

    with torch.no_grad():

        for batch in tqdm(eval_loader):

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            batch_size = batch['input_ids'].size(0)

            total_samples += batch_size

            start_time = time.time()

            outputs = model(**batch)

            end_time = time.time()

            total_latency += (end_time - start_time)

            if args.task == "stsb":

                predictions = outputs.logits.squeeze()

                metric.add_batch(
                    predictions=predictions,
                    references=batch['labels']
                )

            else:

                predictions = torch.argmax(
                    outputs.logits,
                    dim=-1
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )

                all_labels.extend(
                    batch['labels'].cpu().numpy()
                )

                metric.add_batch(
                    predictions=predictions,
                    references=batch['labels']
                )

    result = metric.compute()

    # Add accuracy for all classification datasets
    if args.task != "stsb":

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        result["accuracy"] = accuracy

    avg_latency = total_latency / len(eval_loader)

    throughput = total_samples / total_latency

    print(f"\nValidation Result: {result}")

    print(f"Average Latency: {avg_latency:.6f} sec")

    print(f"Throughput: {throughput:.2f} samples/sec")

    return result, avg_latency, throughput

# =========================================================
# SAVE CHECKPOINT
# =========================================================

def save_checkpoint(epoch, score):

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_file = os.path.join(
        args.save_path,
        f"{args.task}_best.pt"
    )

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'score': score
    }, save_file)

    print(f"\nSaved best model to {save_file}")


# =========================================================
# CSV LOGGING
# =========================================================

results_list = []


# =========================================================
# TRAIN LOOP
# =========================================================

best_score = -float('inf')

print("\nStarting Training...\n")

for epoch in range(1, args.epochs + 1):

    print(f"\n================ EPOCH {epoch} ================\n")

    train_loss = train(epoch)

    result, latency, throughput = validate()

    if "accuracy" in result:
       score = result["accuracy"]

    elif "pearson" in result:
        score = result["pearson"]

    elif "matthews_correlation" in result:
        score = result["matthews_correlation"]

    else:
        score = list(result.values())[0]
    print(f"\nValidation Score: {score:.4f}")

    results_list.append({
        "task": args.task,
        "model": args.model,
        "epoch": epoch,
        "weight_bit": args.weight_bit,
        "activation_bit": args.activation_bit,
        "train_loss": train_loss,
        "accuracy": score,
        "model_size_mb": model_size,
        "latency_sec": latency,
        "throughput": throughput,
        "energy_consumption": energy_consumption
    })

    if score > best_score:

        best_score = score

        save_checkpoint(epoch, score)


# =========================================================
# SAVE CSV
# =========================================================

df = pd.DataFrame(results_list)

df.to_csv(args.csv_path, index=False)

print(f"\nResults Saved To: {args.csv_path}")


# =========================================================
# FINAL RESULTS
# =========================================================

print("\n================ FINAL RESULTS ================\n")

print(f"Model Name: {args.model}")

print(f"Dataset: {args.task}")

print(f"Memory Size: {model_size:.2f} MB")

print(f"Latency: {latency:.6f} sec")

print(f"Accuracy: {best_score*100:.2f}%")

print(f"Bits: Mixed (8/4)")

print(f"Energy Consumption: {energy_consumption}")

print(f"Throughput: {throughput:.2f} samples/sec")

print("\nTraining Finished")