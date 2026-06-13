<p align="center">
  <img src="imgs/hawq_glue_pipeline.png" width="900">
  <br>
  <br>
</p>

# HAWQ-GLUE: Hessian-Aware Mixed-Precision Quantization of Transformer Models on GLUE Benchmarks

PyTorch implementation of HAWQ-style mixed-precision quantization for Transformer models with comprehensive evaluation on GLUE benchmark datasets.

This project provides an end-to-end framework for:

* FP32 baseline training
* HAWQ mixed-precision quantization
* Multi-model support
* Multi-dataset evaluation
* Latency, throughput and energy analysis
* Automatic FP32 vs HAWQ comparison

---

## Features

* FP32 Transformer Training
* HAWQ Mixed Precision Quantization
* INT8 Attention Layers
* INT6 Feed Forward Layers
* FP32 Classification Head
* Automatic Checkpoint Saving
* Mixed Precision Training (AMP)
* Warmup Scheduler + AdamW
* Latency Measurement
* Throughput Measurement
* GPU Energy Consumption Monitoring
* FP32 vs HAWQ Comparison Tables

---

## Supported Models

| Model      |
| ---------- |
| BERT-base  |
| DistilBERT |
| ALBERT     |
| MobileBERT |
| TinyBERT   |
| GPT2       |

---

## Supported GLUE Datasets

| Dataset |
| ------- |
| SST2    |
| QNLI    |
| MNLI    |
| QQP     |
| RTE     |
| MRPC    |
| WNLI    |
| STS-B   |
| CoLA    |

---

## Pipeline

```text
FP32 Training
      ↓
Best Checkpoint Saving
      ↓
Load Fine-tuned Model
      ↓
HAWQ Mixed Precision Quantization
(INT8 Attention + INT6 Feed Forward)
      ↓
Evaluation
      ↓
Accuracy + Latency + Throughput + Energy
```

---

## Installation

### Requirements

* Python ≥ 3.10
* PyTorch ≥ 2.0
* CUDA-enabled GPU
* Transformers
* Datasets
* Evaluate

### Clone Repository

```bash
git clone https://github.com/<username>/HAWQ-GLUE.git

cd HAWQ-GLUE

python -m venv hawq_glue

source hawq_glue/bin/activate

pip install -r requirements.txt
```

---

## Getting Started

### BERT-base + SST2

```bash
python train_glue_hawq.py \
--task sst2 \
--model bert-base-uncased \
--batch-size 16 \
--epochs 3 \
--lr 3e-5
```

### GPT2 + MRPC

```bash
python train_glue_hawq.py \
--task mrpc \
--model gpt2 \
--batch-size 16 \
--epochs 3 \
--lr 3e-5
```

### DistilBERT + QQP

```bash
python train_glue_hawq.py \
--task qqp \
--model distilbert-base-uncased \
--batch-size 16 \
--epochs 3 \
--lr 3e-5
```

---

## Evaluation Metrics

The framework automatically computes:

* Accuracy
* Precision
* Recall
* F1 Score
* Model Size (MB)
* Latency (sec)
* Throughput (samples/sec)
* Energy Consumption

---

## Experimental Results

Results were obtained using multiple Transformer architectures across GLUE benchmark tasks.

### Example Comparison

| Model      | Precision   | Model Size (MB) | Accuracy (%) | Latency (sec) | Throughput |
| ---------- | ----------- | --------------: | -----------: | ------------: | ---------: |
| BERT-base  | FP32        |          417.66 |        xx.xx |        xx.xxx |        xxx |
| BERT-base  | Mixed (8/6) |           xx.xx |        xx.xx |        xx.xxx |        xxx |
| DistilBERT | FP32        |          255.42 |        xx.xx |        xx.xxx |        xxx |
| DistilBERT | Mixed (8/6) |           xx.xx |        xx.xx |        xx.xxx |        xxx |
| TinyBERT   | FP32        |           54.72 |        xx.xx |        xx.xxx |        xxx |
| TinyBERT   | Mixed (8/6) |           xx.xx |        xx.xx |        xx.xxx |        xxx |

Detailed results are available in the generated CSV files.

---

## Repository Structure

```text
HAWQ-GLUE
│
├── checkpoints/
├── imgs/
├── results/
├── train_glue_hawq.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Technologies Used

* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* Evaluate
* CUDA
* Automatic Mixed Precision (AMP)
* Pandas

---

## Key Highlights

✓ FP32 baseline training

✓ HAWQ-style mixed precision quantization

✓ Multi-model support

✓ Multi-dataset GLUE evaluation

✓ INT8/INT6 mixed precision

✓ Energy-aware inference analysis

✓ Latency and throughput benchmarking

✓ Automatic comparison tables

---

## Ongoing Research Directions

* HAWQ-V2 Trace-weighted Quantization
* INT4 Mixed Precision Quantization
* Quantization-Aware Training (QAT)
* ONNX Export
* TensorRT Deployment
* Edge AI Optimization
* Large Language Model Compression

---

## Research Background

This implementation is inspired by the following works:

* HAWQ: Hessian AWare Quantization of Neural Networks with Mixed-Precision (ICCV 2019)
* HAWQ-V2: Hessian Aware Trace-Weighted Quantization of Neural Networks (NeurIPS 2020)
* HAWQ-V3: Dyadic Neural Network Quantization (ICML 2021)

The repository extends these ideas to Transformer architectures and GLUE benchmark datasets using PyTorch and Hugging Face Transformers.

---

## License

For educational and research purposes only.

