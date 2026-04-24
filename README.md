📘 Concept Bottleneck Models on Fashion-MNIST
📌 Overview

This project studies Concept Bottleneck Models (CBMs) and related architectures on the Fashion-MNIST dataset. The goal is to analyze the trade-offs between:

Predictive performance
Interpretability
Steerability

We compare three model families:

Baseline classifier: x→y
Concept Bottleneck Model (CBM): x→c→y
Hybrid CBM with side channel: y=f(c)+s(x)
📊 Dataset

We use Fashion-MNIST, a dataset of grayscale clothing images with 10 classes.

👕 Classes
T-shirt/top, Trouser, Pullover, Dress, Coat
Sandal, Shirt, Sneaker, Bag, Ankle boot
🧠 Concept Definitions

We define 6 binary concepts derived from class labels:

Upper body clothing
Lower body clothing
Footwear
Has sleeves
Open footwear
Accessory (bag)

Concept labels are generated deterministically from class labels.

🏗️ Models Implemented
1️⃣ Baseline Model (x → y)
CNN backbone
Single classification head
Trained with cross-entropy loss
2️⃣ Concept Predictor (x → c)
Same CNN backbone
Multi-label output (6 concepts)
Trained with binary cross-entropy loss
3️⃣ Concept Bottleneck Model (CBM)
Pipeline: image → concepts → label
Independent training:
Train concept model first
Freeze it
Train label predictor on concepts
4️⃣ Hybrid CBM
Combines:
Concept-based prediction f(c)
Direct image prediction s(x)

Final output:

y=f(c)+s(x)
🧪 Experiments
🔹 1. Model Comparison

We evaluate all models using:

Accuracy
AUROC (multiclass, one-vs-rest)
🔹 2. Concept Prediction Evaluation

We evaluate concept quality using:

F1 score per concept
Macro-average F1
🔹 3. Side-Channel Dropout Experiment

We train the Hybrid CBM with dropout probabilities:

p ∈ {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}

We measure:

Accuracy vs dropout
AUROC vs dropout

This analyzes how reliance shifts between:

raw image features
concept bottleneck
🔹 4. Concept Intervention Experiment

We evaluate steerability by:

Predicting concepts for test images
Flipping one concept at a time
Recomputing predictions

We measure:

Average probability change
Label flip rate

Concepts are ranked by their influence on predictions.

📈 Results

The project produces:

Accuracy and AUROC for all models
Concept F1 scores
Dropout vs performance plots
Concept intervention analysis
🧠 Key Findings
Baseline achieves highest accuracy but lacks interpretability
CBM is fully interpretable but slightly less accurate
Hybrid CBM balances performance and interpretability
Increasing dropout:
forces reliance on concepts
improves steerability
may reduce accuracy
▶️ How to Run

Open the notebook:

cbm_fashion_mnist.ipynb

Run all cells:

Kernel → Restart & Run All
Outputs:
Printed metrics (accuracy, AUROC, F1)
Dropout performance plots
Intervention results
📁 Project Structure
.
├── cbm_fashion_mnist.ipynb   # Main implementation
├── README.md                 # Project description
└── report.pdf                # Final report (2–3 pages)
⚙️ Requirements
Python 3.x
PyTorch
torchvision
scikit-learn
matplotlib
numpy

Install dependencies:

pip install torch torchvision scikit-learn matplotlib numpy
📌 Notes
Training is lightweight (3–5 epochs per model)
CPU execution is sufficient
Results may vary slightly due to randomness
