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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
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
parser = argparse.ArgumentParser(description='GLUE Fine Tuning')

parser.add_argument('--task',
                    type=str,
                    default='sst2',
                    choices=['sst2','qnli','mnli','qqp','rte','mrpc','wnli','stsb','cola'])

parser.add_argument('--model',
                    type=str,
                    default='bert-base-uncased')

parser.add_argument('--epochs',
                    type=int,
                    default=3)

parser.add_argument('--batch-size',
                    type=int,
                    default=8)

parser.add_argument('--lr',
                    type=float,
                    default=3e-5)

parser.add_argument('--max-length',
                    type=int,
                    default=256)

parser.add_argument('--seed',
                    type=int,
                    default=42)

parser.add_argument('--save-path',
                    type=str,
                    default='./checkpoints/')

parser.add_argument('--csv-path',
                    type=str,
                    default='./baseline_results.csv')

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("\nUsing device:",device)

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
    'sst2':2,
    'qnli':2,
    'qqp':2,
    'mnli':3,
    'rte':2,
    'mrpc':2,
    'wnli':2,
    'cola':2,
    'stsb':1
}

# =========================================================
# LOAD DATASET
# =========================================================
print("Loading Dataset")

dataset = load_dataset("glue",args.task)

sentence1_key,sentence2_key = task_to_keys[args.task]

# =========================================================
# TOKENIZER
# =========================================================
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

tokenized_dataset = dataset.map(
    preprocess_function,
    batched=True
)

# =========================================================
# DATASETS
# =========================================================
train_dataset = tokenized_dataset['train']

if args.task == "mnli":
    eval_dataset = tokenized_dataset['validation_matched']
else:
    eval_dataset = tokenized_dataset['validation']

train_dataset = train_dataset.rename_column("label","labels")
eval_dataset = eval_dataset.rename_column("label","labels")

train_dataset.set_format(
    type='torch',
    columns=['input_ids','attention_mask','labels']
)

eval_dataset.set_format(
    type='torch',
    columns=['input_ids','attention_mask','labels']
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

if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

model = model.to(device)

# =========================================================
# MODEL SIZE
# =========================================================
def get_model_size(model):

    param_size = 0

    for param in model.parameters():
        param_size += param.nelement()*param.element_size()

    buffer_size = 0

    for buffer in model.buffers():
        buffer_size += buffer.nelement()*buffer.element_size()

    return (param_size+buffer_size)/(1024**2)

model_size = get_model_size(model)

# =========================================================
# GPU POWER
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
# OPTIMIZER
# =========================================================
optimizer = optim.AdamW(
    model.parameters(),
    lr=args.lr
)

total_steps = len(train_loader)*args.epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1*total_steps),
    num_training_steps=total_steps
)

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
            k:v.to(device)
            for k,v in batch.items()
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

    return total_loss/len(train_loader)

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

            batch_size = batch["input_ids"].size(0)

            total_samples += batch_size

            start_time = time.time()

            outputs = model(**batch)

            end_time = time.time()

            total_latency += (end_time - start_time)

            if args.task == "stsb":

                predictions = outputs.logits.squeeze()

                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"]
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
                    batch["labels"].cpu().numpy()
                )

                metric.add_batch(
                    predictions=predictions,
                    references=batch["labels"]
                )

    result = metric.compute()

    if args.task != "stsb":

        accuracy = accuracy_score(
            all_labels,
            all_predictions
        )

        precision = precision_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )

        f1 = f1_score(
            all_labels,
            all_predictions,
            average="weighted",
            zero_division=0
        )

    else:

        accuracy = result["pearson"]
        precision = result["pearson"]
        recall = result["pearson"]
        f1 = result["pearson"]

    avg_latency = total_latency / len(eval_loader)

    throughput = total_samples / total_latency

    energy_consumption = get_gpu_power()

    return result, accuracy, precision, recall, f1, avg_latency, throughput, energy_consumption


# =========================================================
# SAVE CHECKPOINT
# =========================================================
def save_checkpoint(epoch,score):

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    save_file = os.path.join(
        args.save_path,
        f"{args.model.replace('/','_')}_{args.task}_fp32_best.pt"
    )

    torch.save(
        {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "score":score
        },
        save_file
    )

    print("\nSaved Best Model :",save_file)


# =========================================================
# CSV LOGGING
# =========================================================
results_list = []


# =========================================================
# TRAIN LOOP
# =========================================================
best_score = -float("inf")

print("\nStarting Training\n")

for epoch in range(1,args.epochs+1):

    print(f"\n========== EPOCH {epoch} ==========\n")

    train_loss = train(epoch)

    result,accuracy,precision,recall,f1,latency,throughput,energy_consumption = validate()

    score = accuracy

    print("\nAccuracy :",accuracy)
    print("F1 Score :",f1)

    results_list.append({

        "task":args.task,
        "model":args.model,
        "epoch":epoch,
        "train_loss":train_loss,
        "accuracy":accuracy,
        "precision":precision,
        "recall":recall,
        "f1_score":f1,
        "memory_size_mb":model_size,
        "latency_sec":latency,
        "energy_consumption_j":energy_consumption,
        "throughput_samples_sec":throughput,
        "precision_type":"FP32"

    })

    if score > best_score:

        best_score = score

        save_checkpoint(
            epoch,
            score
        )


# =========================================================
# SAVE BASELINE CSV
# =========================================================
df = pd.DataFrame(results_list)

df.to_csv(
    f"results/baseline_{args.model.replace('/','_')}_{args.task}.csv",
    index=False
)

print("\nBaseline Results Saved")


# =========================================================
# FINAL FP32 RESULTS
# =========================================================
print("\n============== FP32 BASELINE ==============\n")

print("Model Name :", args.model)
print("Dataset Name :", args.task)
print("Memory Size (MB) :", round(model_size,2))
print("Latency (sec) :", round(latency,6))
print("Accuracy (%) :", round(best_score*100,2))
print("Bits :", 32)
print("Energy Consumption (J) :", round(energy_consumption,2))
print("Throughput (samples/sec) :", round(throughput,2))
print("Precision :", round(precision,4))
print("Recall :", round(recall,4))
print("F1 Score :", round(f1,4))

# =========================================================
# ACTIVATION QUANTIZATION
# =========================================================
class QuantAct(nn.Module):

    def __init__(self,activation_bit=8):
        super().__init__()
        self.activation_bit = activation_bit

    def forward(self,x):

        qmin = -(2**(self.activation_bit-1))
        qmax = (2**(self.activation_bit-1))-1

        max_val = x.abs().max()

        scale = max_val/qmax

        if scale == 0:
            return x

        x_int = torch.clamp(
            (x/scale).round(),
            qmin,
            qmax
        )

        return x_int*scale


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

    def quantize_weight(self,x):

        qmin = -(2**(self.weight_bit-1))
        qmax = (2**(self.weight_bit-1))-1

        max_val = x.abs().max()

        scale = max_val/qmax

        if scale == 0:
            return x

        x_int = torch.clamp(
            (x/scale).round(),
            qmin,
            qmax
        )

        return x_int*scale

    def forward(self,x):

        if self.training:
            return self.layer(x)

        x = self.activation_quant(x)

        quant_weight = self.quantize_weight(
            self.layer.weight
        )

        return nn.functional.linear(
            x,
            quant_weight,
            self.layer.bias
        )


# =========================================================
# IMPROVED HAWQ MIXED PRECISION
# =========================================================
def quantize_model(module,prefix=""):

    for name,child in module.named_children():

        full_name = prefix + "." + name if prefix else name

        if isinstance(child,nn.Linear):

            # classifier remains FP32
            if "classifier" in full_name:
                continue

            bit = 8

            # attention layers
            if "attention" in full_name:
                bit = 8

            # feedforward intermediate layer
            elif "intermediate.dense" in full_name:
                bit = 6

            # output projection
            elif "output.dense" in full_name:
                bit = 8

            setattr(
                module,
                name,
                QuantLinear(
                    child,
                    weight_bit=bit,
                    activation_bit=8
                )
            )

        else:
            quantize_model(
                child,
                full_name
            )


# =========================================================
# LOAD FINE-TUNED CHECKPOINT
# =========================================================
checkpoint = torch.load(
    os.path.join(
        args.save_path,
        f"{args.model.replace('/','_')}_{args.task}_fp32_best.pt"
    ),
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("\nLoaded Fine-Tuned FP32 Model")


# =========================================================
# APPLY HAWQ
# =========================================================
quantize_model(model)

model = model.to(device)

quant_model_size = get_model_size(model)

print("\nImproved HAWQ Quantization Applied")


# =========================================================
# EVALUATE QUANTIZED MODEL
# =========================================================
result_q,accuracy_q,precision_q,recall_q,f1_q,latency_q,throughput_q,energy_q = validate()


# =========================================================
# FINAL HAWQ RESULTS
# =========================================================
print("\n============== HAWQ RESULTS ==============\n")

print("Model Name :", args.model)
print("Dataset Name :", args.task)
print("Memory Size (MB) :", round(quant_model_size,2))
print("Latency (sec) :", round(latency_q,6))
print("Accuracy (%) :", round(accuracy_q*100,2))
print("Bits : Mixed(8/6)")
print("Energy Consumption (J) :", round(energy_q,2))
print("Throughput (samples/sec) :", round(throughput_q,2))
print("Precision :", round(precision_q,4))
print("Recall :", round(recall_q,4))
print("F1 Score :", round(f1_q,4))


# =========================================================
# COMPARISON TABLE
# =========================================================
comparison = pd.DataFrame({

    "Model Name":[args.model,args.model],

    "Memory Size (MB)":[
        model_size,
        quant_model_size
    ],

    "Latency (sec)":[
        latency,
        latency_q
    ],

    "Accuracy (%)":[
        best_score*100,
        accuracy_q*100
    ],

    "Precision":[
        precision,
        precision_q
    ],

    "Recall":[
        recall,
        recall_q
    ],

    "F1 Score":[
        f1,
        f1_q
    ],

    "Bits":[
        32,
        "Mixed(8/6)"
    ],

    "Energy Consumption (J)":[
        energy_consumption,
        energy_q
    ],

    "Throughput (samples/sec)":[
        throughput,
        throughput_q
    ],

    "Precision Type":[
        "FP32",
        "INT8/INT6"
    ]

},
index=["FP32","HAWQ"])

comparison.to_csv(
    f"results/comparison_{args.model.replace('/','_')}_{args.task}.csv"
)

print("\n============== COMPARISON TABLE ==============\n")
print(comparison)

