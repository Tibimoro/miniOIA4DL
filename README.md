# miniOIA4DL – Optimized Deep Learning Framework for OIA

This repository contains my optimized version of **miniOIA4DL**, a deep learning framework used in the subject **Optimización de entornos software para IA (OIA)**.

The objective of this work was to identify the main performance bottleneck of the framework and progressively apply optimization techniques to improve inference speed on **TinyCNN with CIFAR-100**.

---

## 🚀 Implemented Optimizations

The following optimization strategies were implemented in the `Conv2D` layer:

- **conv_algo 0 → Baseline direct convolution**
- **conv_algo 1 → im2col + matrix multiplication**
- **conv_algo 2 → Numba JIT acceleration**
- **conv_algo 3 → Cython implementation**
- **conv_algo 4 → Blocked GEMM + cache-aware optimization**

The best final version is:

```bash
python main.py --model TinyCNN --batch_size 8 --performance --conv_algo 4
```

---

## 📊 Best Performance Achieved

### Baseline
- Total time: **150.30 s**
- IPS: **0.05 images/s**

### Best optimized version
- Total time: **0.12 s**
- IPS: **68.96 images/s**

This represents a speedup of more than **1300x** compared with the original implementation.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/Tibimoro/miniOIA4DL.git
cd miniOIA4DL
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build Cython modules:

```bash
cd cython_modules
python setup.py build_ext --inplace
cd ..
```

---

## ▶ Usage

Main arguments:

- `--model` → AlexNet, TinyCNN, OIANet, ResNet18
- `--batch_size` → batch size
- `--performance` → performance profiling
- `--conv_algo` → convolution algorithm

### Supported convolution algorithms

- `0` → direct
- `1` → im2col
- `2` → Numba
- `3` → Cython
- `4` → blocked GEMM

### Example

```bash
python main.py --model TinyCNN --batch_size 8 --performance --conv_algo 4
```

---

## 📁 Repository Structure

```text
miniOIA4DL/
├── cython_modules/
├── data/
├── models/
├── modules/
├── unit_tests/
├── main.py
├── performance.py
├── train.py
├── eval.py
├── requirements.txt
└── README.md
```

---

## 🎯 Academic Context

This repository was developed as part of the coursework for:

**Máster Universitario en Computación en la Nube y de Altas Prestaciones**  
**Asignatura: Optimización de entornos software para IA (OIA)**  
**Curso 2025–2026**
