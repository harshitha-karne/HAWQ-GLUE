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
# Experimental Results

## SST2

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput (samples/sec) | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.402999 | 79.01 | Mixed(8/6) | 251.35 | 54.56 | 0.8111 | 0.7901 | 0.7872 |
| DistilBERT | 255.42 | 0.559558 | 89.56 | Mixed(8/6) | 235.34 | 55.65 | 0.8978 | 0.8956 | 0.8964 |
| ALBERT | 44.58 | 0.998966 | 51.03 | Mixed(8/6) | 241.66 | 31.18 | 0.5323 | 0.5103 | 0.3541 |
| MobileBERT | 93.78 | 3.649336 | 49.08 | Mixed(8/6) | 257.46 | 8.53 | 0.2409 | 0.4908 | 0.3232 |
| GPT2 | 486.71 | 0.028116 | 91.74 | Mixed(8/6) | 239.71 | 562.90 | 0.9176 | 0.9174 | 0.9174 |
| BERT-base | 417.66 | 1.177184 | 91.63 | Mixed(8/6) | 248.96 | 13.47 | 0.9164 | 0.9163 | 0.9163 |

---

## QNLI

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.377278 | 72.84 | Mixed(8/6) | 236.97 | 168.37 | 0.7586 | 0.7284 | 0.7209 |
| DistilBERT | 255.42 | 0.488661 | 84.50 | Mixed(8/6) | 246.60 | 65.38 | 0.8465 | 0.8450 | 0.8447 |
| ALBERT | 44.58 | 0.882888 | 50.63 | Mixed(8/6) | 243.88 | 36.19 | 0.5059 | 0.5063 | 0.5051 |
| MobileBERT | 93.78 | 2.680313 | 49.46 | Mixed(8/6) | 273.90 | 11.92 | 0.2446 | 0.4946 | 0.3274 |
| GPT2 | 486.71 | 0.054777 | 86.03 | Mixed(8/6) | 240.66 | 291.61 | 0.8666 | 0.8603 | 0.8599 |
| BERT-base | 417.66 | 0.945444 | 86.12 | Mixed(8/6) | 242.13 | 16.90 | 0.8695 | 0.8612 | 0.8603 |

---

## MNLI

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.193767 | 61.17 | Mixed(8/6) | 238.11 | 328.92 | 0.6410 | 0.6117 | 0.6114 |
| DistilBERT | 255.42 | 0.533238 | 77.73 | Mixed(8/6) | 259.86 | 59.96 | 0.7847 | 0.7773 | 0.7785 |
| ALBERT | 44.58 | 0.824592 | 34.22 | Mixed(8/6) | 247.89 | 38.77 | 0.3374 | 0.3422 | 0.3828 |
| MobileBERT | 93.78 | 3.046205 | 33.74 | Mixed(8/6) | 245.85 | 10.50 | 0.1072 | 0.3274 | 0.1615 |
| GPT2 | 486.71 | 0.064670 | 80.73 | Mixed(8/6) | 227.35 | 247.18 | 0.8085 | 0.8073 | 0.8077 |
| BERT-base | 417.66 | 0.720849 | 67.68 | Mixed(8/6) | 235.80 | 22.18 | 0.7871 | 0.6768 | 0.6697 |

---

## QQP

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.456795 | 83.43 | Mixed(8/6) | 237.16 | 140.04 | 0.8520 | 0.8343 | 0.8370 |
| DistilBERT | 255.42 | 0.459731 | 85.96 | Mixed(8/6) | 265.95 | 69.57 | 0.8590 | 0.8596 | 0.8576 |
| ALBERT | 44.58 | 0.733726 | 42.25 | Mixed(8/6) | 258.41 | 43.59 | 0.5553 | 0.4225 | 0.3669 |
| MobileBERT | 93.78 | 2.910161 | 60.56 | Mixed(8/6) | 247.28 | 10.99 | 0.5330 | 0.6056 | 0.5252 |
| GPT2 | 486.71 | 0.029125 | 88.78 | Mixed(8/6) | 253.63 | 549.32 | 0.8896 | 0.8878 | 0.8883 |
| BERT-base | 417.66 | 0.991793 | 66.70 | Mixed(8/6) | 261.93 | 16.13 | 0.8041 | 0.6670 | 0.6620 |

---

## RTE

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.489328 | 63.18 | Mixed(8/6) | 248.57 | 62.90 | 0.6631 | 0.6318 | 0.6215 |
| DistilBERT | 255.42 | 0.499146 | 57.40 | Mixed(8/6) | 251.12 | 30.83 | 0.5751 | 0.5740 | 0.5563 |
| ALBERT | 44.58 | 1.284073 | 56.07 | Mixed(8/6) | 240.51 | 11.98 | 0.7517 | 0.5307 | 0.3719 |
| MobileBERT | 93.78 | 6.507542 | 53.79 | Mixed(8/6) | 242.54 | 2.36 | 0.5436 | 0.5379 | 0.4504 |
| GPT2 | 486.71 | 0.050793 | 65.34 | Mixed(8/6) | 231.25 | 155.81 | 0.6645 | 0.6534 | 0.6411 |
| BERT-base | 417.66 | 1.123085 | 57.76 | Mixed(8/6) | 248.49 | 7.05 | 0.6080 | 0.5776 | 0.5211 |

---

## MRPC

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.635653 | 78.43 | Mixed(8/6) | 247.96 | 49.37 | 0.8098 | 0.7843 | 0.7903 |
| DistilBERT | 255.42 | 0.723482 | 70.83 | Mixed(8/6) | 234.63 | 21.69 | 0.7699 | 0.7083 | 0.6126 |
| ALBERT | 44.58 | 1.425603 | 63.97 | Mixed(8/6) | 261.82 | 11.01 | 0.5370 | 0.6397 | 0.5609 |
| MobileBERT | 93.78 | 5.236038 | 35.78 | Mixed(8/6) | 220.69 | 3.00 | 0.5776 | 0.3578 | 0.2759 |
| GPT2 | 486.71 | 0.038823 | 80.88 | Mixed(8/6) | 234.65 | 206.06 | 0.8037 | 0.8088 | 0.8018 |
| BERT-base | 417.66 | 0.918081 | 74.51 | Mixed(8/6) | 218.61 | 8.71 | 0.7862 | 0.7451 | 0.6861 |

...

---

## WNLI

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.600601 | 56.34 | Mixed(8/6) | 237.51 | 23.64 | 0.3174 | 0.5634 | 0.4060 |
| DistilBERT | 255.42 | 0.652551 | 56.34 | Mixed(8/6) | 252.35 | 12.09 | 0.3174 | 0.5634 | 0.4060 |
| ALBERT | 44.58 | 1.445390 | 59.15 | Mixed(8/6) | 266.85 | 5.46 | 0.6204 | 0.5915 | 0.5022 |
| MobileBERT | 93.78 | 6.263815 | 60.56 | Mixed(8/6) | 241.01 | 1.26 | 0.6040 | 0.6056 | 0.5699 |
| GPT2 | 486.71 | 0.030109 | 56.34 | Mixed(8/6) | 239.26 | 131.01 | 0.3174 | 0.5634 | 0.4060 |
| BERT-base | 417.66 | 1.374837 | 39.44 | Mixed(8/6) | 241.76 | 2.87 | 0.4031 | 0.3944 | 0.3968 |

---

## STS-B

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.498826 | 83.33 | Mixed(8/6) | 240.96 | 63.98 | 0.8333 | 0.8333 | 0.8333 |
| DistilBERT | 255.42 | 0.536030 | 57.67 | Mixed(8/6) | 244.79 | 29.77 | 0.5767 | 0.5767 | 0.5767 |
| ALBERT | 44.58 | 1.076902 | 27.28 | Mixed(8/6) | 235.64 | 14.82 | 0.2728 | 0.2728 | 0.2728 |
| MobileBERT | 93.78 | 6.754137 | 9.54 | Mixed(8/6) | 233.33 | 2.36 | 0.0954 | 0.0954 | 0.0954 |
| GPT2 | 486.71 | 0.026446 | 86.45 | Mixed(8/6) | 244.53 | 301.70 | 0.8645 | 0.8645 | 0.8645 |
| BERT-base | 417.66 | 0.888101 | 53.90 | Mixed(8/6) | 245.18 | 8.98 | 0.5390 | 0.5390 | 0.5390 |

---

## CoLA

| Model | Memory Size (MB) | Latency (sec) | Accuracy (%) | Bits | Energy (J) | Throughput | Precision | Recall | F1 Score |
|---------|---------:|---------:|---------:|---------|---------:|---------:|---------:|---------:|---------:|
| TinyBERT | 54.72 | 0.465895 | 69.13 | Mixed(8/6) | 217.41 | 67.84 | 0.4779 | 0.6913 | 0.5651 |
| DistilBERT | 255.42 | 0.597994 | 75.84 | Mixed(8/6) | 247.08 | 26.43 | 0.7574 | 0.7584 | 0.7237 |
| ALBERT | 44.58 | 1.441449 | 40.27 | Mixed(8/6) | 240.95 | 10.96 | 0.6412 | 0.4027 | 0.3573 |
| MobileBERT | 93.78 | 6.375986 | 33.27 | Mixed(8/6) | 247.68 | 2.48 | 0.5698 | 0.3327 | 0.2239 |
| GPT2 | 486.71 | 0.023404 | 79.29 | Mixed(8/6) | 220.21 | 340.19 | 0.7998 | 0.7929 | 0.7768 |
| BERT-base | 417.66 | 1.322375 | 35.38 | Mixed(8/6) | 248.73 | 6.02 | 0.7110 | 0.3538 | 0.2446 |

---

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

