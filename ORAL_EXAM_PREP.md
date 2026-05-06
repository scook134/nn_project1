# ORAL EXAM PREP

This is an exam guide for this specific repository. It is grounded mainly in `project1.ipynb`, with supporting numbers from `report.md` and the stored Fashion-MNIST files under `data/`. If something is not directly shown by the repo, it is labeled `unclear from repo`.

# 1. Project Overview

## What problem this project solves

This project studies an interpretability-versus-accuracy question:

- can we classify Fashion-MNIST well with a standard CNN?
- what happens if we force the model to pass through explicit human-readable concepts first?
- can a hybrid model recover most of the lost accuracy while keeping an interpretable concept interface?

The core comparison in `project1.ipynb` is between:

- `BaselineModel`: direct image-to-class classifier
- `ConceptModel`: image-to-concepts predictor
- `CBM`: concept bottleneck model, image-to-concepts-to-class
- `HybridCBM`: concept path plus direct side channel

## What dataset/task it uses

In `project1.ipynb` cell 3, the repo loads Fashion-MNIST with `torchvision.datasets.FashionMNIST(root='./data', train=..., download=True, transform=transform)`.

The main task is:

- input: a grayscale `28 x 28` clothing image
- output: one of 10 clothing classes

The repo adds a second supervised target:

- an 8-dimensional binary concept vector created by `get_concepts(label)` in cell 5

Important detail:

- the concept labels are not manually annotated from images
- they are deterministic functions of the class label

## What the input and output are

After `ConceptDataset` in cell 7, each sample is:

- `x`: image tensor with shape `[1, 28, 28]`
- `y`: class label scalar
- `c`: concept vector with shape `[8]`

With a `DataLoader(batch_size=64)`, a typical batch is:

- `x`: `[64, 1, 28, 28]`
- `y`: `[64]`
- `c`: `[64, 8]`

Model outputs:

- `BaselineModel.forward(x)`: logits `[B, 10]`
- `ConceptModel.forward(x)`: sigmoid probabilities `[B, 8]`
- `CBM.forward(x)`: logits `[B, 10]`
- `HybridCBM.forward(x)`: tuple `(y, c)` with shapes `[B, 10]` and `[B, 8]`

## What the final model is trying to learn

There is no single final deployed model in the repo. This is an experiment comparing model families.

What each one learns:

- `BaselineModel` learns class-discriminative visual features directly
- `ConceptModel` learns to predict the 8 binary concepts from the image
- `CBM` learns a label classifier on top of predicted concepts
- `HybridCBM` learns both concept prediction and label prediction jointly, while also keeping a direct feature-to-label shortcut

## What the main experimental goal is

The main goal is to test whether concept bottlenecks improve interpretability at an acceptable accuracy cost.

More precisely:

- how much accuracy is lost when class prediction is forced through only 8 concepts
- whether a hybrid architecture recovers most of that lost information
- whether concept predictions are strong enough to be meaningful
- whether changing concept values changes downstream predictions

# 2. Repository Map

## `project1.ipynb`

What it does:

- contains the full pipeline: imports, data loading, concept construction, model classes, training loops, evaluation, dropout sweep, and intervention code

Why it exists:

- this is the real codebase; there are no separate `.py` training modules

How it connects:

- all results in the repo come from this notebook

Functions/classes to know:

- `get_concepts`
- `ConceptDataset`
- `CNNBackbone`
- `BaselineModel`
- `ConceptModel`
- `CBM`
- `HybridCBM`
- `train_baseline`
- `train_concept`
- `evaluate_full`
- `evaluate_concepts`
- `evaluate_concepts_hybrid`
- `predict_from_concepts`
- `intervention_experiment`

## `report.md`

What it does:

- gives the written interpretation of the project and the cleanest results table

Why it exists:

- likely the report submitted with the project

How it connects:

- summarizes the same models implemented in the notebook
- gives hybrid dropout-sweep numbers that are easier to read than the notebook plot output

Important caution:

- exact metrics in `report.md` do not perfectly match the stored notebook outputs
- example: notebook baseline accuracy is `0.9124`, while `report.md` states `0.9080`
- the most likely explanation is different runs, but that is `unclear from repo`

## `dropout_plot.png`

What it does:

- stores the hybrid dropout sweep figure

Why it exists:

- visualizes how accuracy and AUROC vary with side-channel dropout

How it connects:

- corresponds to `results.append((p, acc, auroc))` in cell 23 and the plotting code in cell 24

## `data/FashionMNIST/raw/`

What it does:

- stores the downloaded Fashion-MNIST raw files

Why it exists:

- supports local dataset loading through torchvision

How it connects:

- used by the calls in cell 3

# 3. End-to-End Pipeline

## Step 1: Data loading and preprocessing

In cell 3:

- `transform = transforms.ToTensor()`
- `train_dataset = torchvision.datasets.FashionMNIST(...)`
- `test_dataset = torchvision.datasets.FashionMNIST(...)`

Actual preprocessing in the repo:

- convert PIL image to tensor
- scale pixel values from `[0, 255]` to `[0, 1]`

Not present in the repo:

- no mean/std normalization
- no augmentation
- no train/validation split

## Step 2: Concept construction

Cell 5 defines `get_concepts(label)`.

It returns `torch.zeros(8)` and turns entries on according to the class:

- concept 0: `is_footwear` for classes `[5, 7, 9]`
- concept 1: `is_closed_footwear` for `[7, 9]`
- concept 2: `is_footwear_or_bag` for `[5, 7, 8, 9]`
- concept 3: `has_sleeves` for `[0, 2, 3, 4, 6]`
- concept 4: `has_collar` for `[4, 6]`
- concept 5: `is_long_garment` for `[3, 4]`
- concept 6: `is_outerwear_layer` for `[2, 4]`
- concept 7: `is_legwear_or_footwear` for `[1, 5, 7, 9]`

Important implication:

- concept supervision is perfectly reproducible
- but the concept set is hand-designed and coarse
- because concepts come from labels, not direct human annotation of each image, they do not provide independent ground-truth semantics beyond the class taxonomy

## Step 3: Dataset wrapping

Cell 7 defines `ConceptDataset`.

`__getitem__(idx)` does:

- `x, y = self.dataset[idx]`
- `c = get_concepts(y)`
- `return x, y, c`

This matters because every later training loop expects the loader to return triples `(x, y, c)`, not just `(x, y)`.

Also note:

- in cell 3, `train_loader` and `test_loader` are first built from raw Fashion-MNIST
- in cell 7, those names are overwritten with loaders built from `ConceptDataset`
- all later code uses the concept-augmented loaders

## Step 4: Model construction

All models rely on `CNNBackbone` in cell 9.

Backbone shape flow:

- input: `[B, 1, 28, 28]`
- after `Conv2d(1, 32, 3, padding=1)`: `[B, 32, 28, 28]`
- after `ReLU`: `[B, 32, 28, 28]`
- after `MaxPool2d(2)`: `[B, 32, 14, 14]`
- after `Conv2d(32, 64, 3, padding=1)`: `[B, 64, 14, 14]`
- after `ReLU`: `[B, 64, 14, 14]`
- after second `MaxPool2d(2)`: `[B, 64, 7, 7]`
- after flatten: `[B, 3136]`
- after `Linear(3136, 128)`: `[B, 128]`

This `[B, 128]` vector is the shared hidden representation `h`.

## Step 5: Baseline training loop

Cell 12 defines `train_baseline(model, loader, epochs=5)`.

Actual loop:

1. move model to `device`
2. create `optimizer = optim.Adam(model.parameters(), lr=1e-3)`
3. create `loss_fn = nn.CrossEntropyLoss()`
4. for each epoch:
5. call `model.train()`
6. for each batch `(x, y, _)`:
7. move `x` and `y` to device
8. `optimizer.zero_grad()`
9. `logits = model(x)`
10. `loss = loss_fn(logits, y)`
11. `loss.backward()`
12. `optimizer.step()`
13. accumulate `total_loss += loss.item()`

Important precision:

- `total_loss` is the sum of batch losses over the epoch, not the mean epoch loss
- because `CrossEntropyLoss()` defaults to mean reduction within a batch, each `loss.item()` is already a batch average
- the printed epoch loss therefore depends slightly on the number of batches

## Step 6: Concept-model training loop

Cell 15 defines `train_concept(model, loader, epochs=5)`.

Actual loop:

- optimizer: `Adam(model.parameters(), lr=1e-3)`
- loss: `nn.BCELoss()`
- batch inputs: `(x, _, c)`
- forward: `preds = model(x)` with shape `[B, 8]`
- target: `c` with shape `[B, 8]`
- backward and optimizer step as usual

Important code detail:

- unlike `train_baseline`, this function does not explicitly call `model.train()`
- however, `concept_model = ConceptModel()` is created immediately before training, and PyTorch modules start in training mode by default
- so the code still behaves as training code, but it is less explicit than it should be

## Step 7: CBM training loop

Cell 19 trains the strict bottleneck model:

- `cbm = CBM(concept_model).to(device)`
- `optimizer = optim.Adam(cbm.classifier.parameters(), lr=1e-3)`
- `loss_fn = nn.CrossEntropyLoss()`

Important code detail:

- only `cbm.classifier.parameters()` are optimized
- `CBM.forward(x)` computes `c = self.concept_model(x)` inside `with torch.no_grad():`
- so the concept predictor is frozen during CBM classifier training

Another important detail:

- the CBM classifier is trained on continuous concept probabilities in `[0, 1]`
- it is not trained on thresholded binary concepts

That means the CBM is strict in architecture, but its classifier still sees soft concept confidence values, not hard concept decisions.

## Step 8: Hybrid training loop

Cell 23 trains a new `HybridCBM` for each dropout in:

- `[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]`

For each `p`:

1. instantiate `model = HybridCBM(dropout_p=p).to(device)`
2. create `optimizer = optim.Adam(model.parameters(), lr=1e-3)`
3. for 3 epochs:
4. for each batch `(x, y, c)`:
5. `optimizer.zero_grad()`
6. `y_pred, c_pred = model(x)`
7. compute `nn.CrossEntropyLoss()(y_pred, y) + nn.BCELoss()(c_pred, c)`
8. backprop and step
9. evaluate on the test set and append `(p, acc, auroc)` to `results`

Important code details:

- the loop does not explicitly call `model.train()`, but a newly created module defaults to train mode
- the class loss and concept loss are simply added with equal weight `1.0 + 1.0`
- no validation set is used to choose dropout; reported comparisons are all on the test set
- after the loop ends, `model` refers to the last trained hybrid, which is the one with `dropout_p=0.9`

## Step 9: Evaluation

### `evaluate_full` in cell 10

Computes:

- test accuracy
- multiclass AUROC with `roc_auc_score(all_labels, all_probs, multi_class='ovr')`

Important details:

- it applies `torch.softmax(logits, dim=1)` before AUROC
- for hybrid outputs, it uses only `output[0]`, meaning the class logits `y`
- it assumes loader batches have shape `(x, y, _)`

### `evaluate_concepts` in cell 16

Computes:

- per-concept F1
- macro F1

Actual logic:

- predict probabilities
- threshold at `0.5`
- compute `f1_score` separately for each concept dimension
- average the 8 F1 values manually

### `evaluate_concepts_hybrid` in cell 25

Same idea as `evaluate_concepts`, but for the hybrid model:

- calls `_, preds = model(x)`
- thresholds the concept predictions at `0.5`

Important detail:

- this evaluation is run in cell 26 on the variable `model`, which at that point is the last hybrid trained in the dropout sweep, not necessarily the best one

## Step 10: Intervention analysis

Cell 29 defines `intervention_experiment(model, loader, num_concepts=8)`.

What it does:

1. compute original predictions from the model
2. for each concept index `i`:
3. flip that concept with `c_flipped[:, i] = 1 - c_flipped[:, i]`
4. recompute label logits with `predict_from_concepts(model, c_flipped)`
5. measure average absolute probability change
6. measure how often the predicted label changes

Important methodological caveat:

- for the hybrid model, the original logits come from the full model `y = y_c + y_s`
- after intervention, `predict_from_concepts` uses only `label_from_concepts(c_flipped)`
- so the intervention does not isolate "one concept changed while everything else stayed the same"
- it changes the concept and also removes the side channel from the recomputed prediction

This is one of the most important caveats to mention in an oral exam.

## Step 11: Saving/loading outputs

Present in the repo:

- dataset files under `data/`
- plot image `dropout_plot.png`
- notebook outputs stored in `project1.ipynb`
- summary text in `report.md`

Not present:

- no `torch.save(...)` checkpoints
- no saved training histories as separate files
- no reusable CLI training script

# 4. Model Architecture

## `CNNBackbone`

Defined in `project1.ipynb` cell 9.

Architecture:

- `nn.Conv2d(1, 32, 3, padding=1)`
- `nn.ReLU()`
- `nn.MaxPool2d(2)`
- `nn.Conv2d(32, 64, 3, padding=1)`
- `nn.ReLU()`
- `nn.MaxPool2d(2)`
- flatten
- `nn.Linear(64 * 7 * 7, 128)`

Why this makes sense here:

- Fashion-MNIST images are small and grayscale
- CNNs exploit spatial locality better than an MLP on flattened pixels
- the architecture is small enough to train quickly in a notebook

What is not in this backbone:

- no batch normalization
- no dropout
- no residual connections
- no deeper feature hierarchy

## `BaselineModel`

Defined in cell 9.

Computation:

- `h = self.backbone(x)` gives `[B, 128]`
- `self.classifier(h)` gives `[B, 10]`

Interpretation:

- all class information must be encoded directly in `h`
- there is no explicit interpretability layer

## `ConceptModel`

Defined in cell 9.

Computation:

- `h = self.backbone(x)` gives `[B, 128]`
- `self.head(h)` gives raw concept scores `[B, 8]`
- `torch.sigmoid(...)` turns them into concept probabilities `[B, 8]`

Interpretation:

- each concept is predicted independently
- multiple concepts can be active at once

## `CBM`

Defined in cell 9.

Computation:

- `c = self.concept_model(x)` gives `[B, 8]`
- `self.classifier(c)` gives `[B, 10]`

Why this is a real bottleneck:

- the label classifier only sees the 8-dimensional concept vector
- it does not receive the 128-dimensional hidden features directly

What to say carefully:

- this is a stricter bottleneck than the hybrid model
- but it still uses soft concept probabilities, not hard binary concept decisions

## `HybridCBM`

Defined in cell 9.

Computation:

- `h = self.backbone(x)` gives `[B, 128]`
- concept branch:
  - `concept_head(h)` -> `[B, 8]`
  - `sigmoid(...)` -> `c` with shape `[B, 8]`
  - `label_from_concepts(c)` -> `y_c` with shape `[B, 10]`
- side branch:
  - `dropout(h)` -> `[B, 128]`
  - `side_head(...)` -> `y_s` with shape `[B, 10]`
- final logits:
  - `y = y_c + y_s` with shape `[B, 10]`

Why this architecture was chosen:

- the 8 concepts may not be enough to preserve all class information
- the side branch gives the model a way to keep residual information outside the bottleneck

Exact tradeoff:

- better accuracy than a strict bottleneck
- weaker interpretability purity because final decisions do not depend only on concepts

## Activation functions

Actually used in code:

- `ReLU` in the backbone
- `sigmoid` for concept probabilities
- `softmax` only inside evaluation, not inside the model classes

What is not used:

- no softmax in model forward passes for class logits

That is correct because:

- `CrossEntropyLoss` expects raw logits

## Alternatives that are plausible but not implemented

Grounded alternatives:

- replace `BCELoss + sigmoid` with `BCEWithLogitsLoss`
- jointly fine-tune the concept extractor in the strict CBM instead of freezing it
- tune the relative weight between class loss and concept loss in the hybrid
- add normalization or augmentation

Less safe to claim:

- deeper modern architectures would probably help, but the repo does not test that

# 5. Training Details

## Hyperparameters directly visible in code

From cells 3, 12, 15, 19, 23, 16, and 25:

- batch size: `64`
- optimizer everywhere: `optim.Adam`
- learning rate everywhere: `1e-3`
- baseline epochs: `5`
- concept-model epochs: `5`
- CBM classifier epochs: `5`
- hybrid epochs per dropout value: `3`
- hybrid dropout values: `0.0, 0.1, 0.3, 0.5, 0.7, 0.9`
- concept threshold for evaluation: `0.5`

Device:

- cell 1 sets `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- stored notebook output shows `cpu`

## Loss functions

Baseline:

- `nn.CrossEntropyLoss()` on class logits `[B, 10]` and labels `[B]`

Concept model:

- `nn.BCELoss()` on predicted concept probabilities `[B, 8]` and targets `[B, 8]`

CBM:

- `nn.CrossEntropyLoss()` on class logits `[B, 10]`

Hybrid:

- `nn.CrossEntropyLoss()(y_pred, y) + nn.BCELoss()(c_pred, c)`

Important implication:

- the hybrid multi-task objective is unweighted except for the implicit equal coefficient `1`
- the repo does not test whether another weighting would work better

## Metrics

Class metrics:

- accuracy
- one-vs-rest multiclass AUROC

Concept metrics:

- per-concept F1
- macro F1 = mean of the 8 per-concept F1 values

Why these metrics are reasonable:

- accuracy is the standard top-1 metric
- AUROC gives a threshold-independent ranking measure
- F1 is appropriate for binary concept prediction

## Regularization

Actually present:

- dropout only in the hybrid side channel

Not present:

- no weight decay
- no augmentation
- no early stopping
- no scheduler
- no explicit seed control

## What each choice implies

- using Adam with `1e-3` makes the notebook simple and fast
- short training means the experiments are lightweight, but may be under-tuned
- no validation split weakens model-selection discipline
- no seed control means exact numbers are not guaranteed to be reproducible
- equal weighting in the hybrid objective may under- or over-emphasize concept prediction

# 6. Architectural Decisions and Tradeoffs

## Choice: CNN backbone instead of a plain MLP

What the repo chose:

- `CNNBackbone`

Why it makes sense:

- images have 2D structure
- convolutions are a natural inductive bias for local visual patterns

Tradeoff:

- a CNN is still fairly interpretable at the feature-map level, but not as semantically transparent as a concept bottleneck

What could go wrong:

- limited capacity if the task needed more complex features

## Choice: hand-crafted label-derived concepts

What the repo chose:

- `get_concepts(label)` in cell 5

Why it makes sense:

- concept supervision becomes cheap and deterministic

Tradeoff:

- concept quality is limited by the hand-designed taxonomy

What could go wrong:

- classes can share similar concept vectors
- concept labels are not independent of class labels
- some concepts are correlated or partially redundant

What to try next:

- richer or more fine-grained concepts
- human-labeled concepts at image level

## Choice: independent CBM instead of end-to-end CBM

What the repo chose:

- train `ConceptModel` first
- then freeze it inside `CBM`

Why it makes sense:

- it creates a very clear bottleneck
- it is easy to explain and inspect

Tradeoff:

- the concept extractor cannot adapt during label-classifier training

What could go wrong:

- concept prediction errors propagate directly to final classification

## Choice: soft concepts instead of thresholded hard concepts

What the repo chose:

- the CBM classifier receives predicted probabilities from `ConceptModel`

Why it matters:

- this preserves uncertainty information
- a concept score like `0.62` carries more information than thresholding immediately to `1`

Tradeoff:

- slightly less "symbolic" than a hard bottleneck

What could go wrong:

- the classifier may exploit confidence patterns rather than purely binary concept states

## Choice: hybrid side channel

What the repo chose:

- `HybridCBM` adds `y_c + y_s`

Why it makes sense:

- if 8 concepts are not sufficient, the side path can carry missing class information

Tradeoff:

- improved predictive performance
- weaker guarantee that concepts fully explain decisions

What could go wrong:

- the side channel may dominate
- concept predictions may look good but contribute less than expected to final classification

## Choice: dropout sweep on the side channel

What the repo chose:

- test dropout values from `0.0` to `0.9`

Why it makes sense:

- this is a direct way to weaken the shortcut and see whether performance stays high

Tradeoff:

- more dropout may encourage reliance on concepts
- too much dropout may damage the side path too much

What could go wrong:

- with only 3 epochs and no validation split, the sweep is informative but not definitive

# 7. Results and Interpretation

## Result sources

There are two sources:

- stored notebook outputs in `project1.ipynb`
- the summarized table in `report.md`

Be strict in the exam:

- quote the source when giving numbers
- if asked why numbers differ slightly, say `unclear from repo`, most likely different runs

## Notebook outputs

Stored notebook outputs show:

- baseline accuracy: `0.9124`
- baseline AUROC: `0.9940248222222223`
- concept macro F1: `0.9479902873008553`
- CBM accuracy: `0.8073`
- CBM AUROC: `0.9801787611111109`
- CBM concept macro F1: `0.9479902873008553`
- final evaluated hybrid concept macro F1: `0.9455496495738474`

Per-concept F1 from cell 17:

- concept 0: `0.9997`
- concept 1: `0.9902`
- concept 2: `0.9966`
- concept 3: `0.9940`
- concept 4: `0.8412`
- concept 5: `0.8654`
- concept 6: `0.9013`
- concept 7: `0.9956`

What is safe to conclude:

- concept prediction is strong overall
- concepts 4, 5, and 6 are weaker than the others
- the strict CBM is materially worse than the baseline on label prediction

## Results from `report.md`

`report.md` reports:

- baseline CNN: `0.9080` accuracy, `0.9938` AUROC
- independent CBM: `0.8109` accuracy, `0.9824` AUROC
- hybrid `p=0.0`: `0.8981` accuracy, `0.9931` AUROC
- hybrid `p=0.1`: `0.8947` accuracy, `0.9926` AUROC
- hybrid `p=0.3`: `0.9035` accuracy, `0.9931` AUROC
- hybrid `p=0.5`: `0.9001` accuracy, `0.9926` AUROC
- hybrid `p=0.7`: `0.8959` accuracy, `0.9927` AUROC
- hybrid `p=0.9`: `0.8989` accuracy, `0.9917` AUROC

What is safe to conclude from that table:

- baseline is still the top classifier
- the strict bottleneck loses roughly 10 percentage points of accuracy relative to baseline
- the hybrid recovers most of that gap
- the dropout trend is not monotonic

Do not overclaim:

- the repo does not prove that `p=0.3` is globally optimal
- it only shows that in one reported sweep, `p=0.3` was the best among the tested values

## Intervention outputs

Stored notebook outputs for `intervention_experiment` show average probability changes roughly between `0.1305` and `0.1420`, and substantial label flip rates.

Safe interpretation:

- the concept pathway is not numerically inert
- changing concept values affects downstream predictions

Unsafe interpretation:

- you should not say this is a clean causal intervention study, because the implementation mixes concept changes with concept-only recomputation in the hybrid case

## Bottom-line interpretation

The evidence in this repo supports this narrow conclusion:

"On this implementation of Fashion-MNIST, a strict concept bottleneck is more interpretable but less accurate than a baseline CNN, while a hybrid concept-plus-shortcut model recovers much of the lost label performance."

That is stronger than:

- "concept bottlenecks always work well"

and weaker than:

- "the hybrid is fully interpretable"

## Main limitations to say out loud

- no validation split
- no repeated runs across seeds
- no saved checkpoints
- no model-selection protocol beyond the stored sweep
- concepts are derived from labels
- hybrid concept evaluation and intervention are run only on the last trained hybrid instance in the notebook
- intervention code for the hybrid does not preserve the side channel in the recomputation

# 8. Likely Professor Questions

## Core understanding

### 1. What is the scientific question here?

Whether forcing image classification through explicit concepts improves interpretability, and how much label accuracy is lost or recovered by strict and hybrid concept-bottleneck designs.

### 2. Where in the code are the concepts defined?

In `project1.ipynb` cell 5, in the function `get_concepts(label)`.

### 3. Are those concepts learned from human annotation?

No. They are generated deterministically from the class label. That makes them reproducible but not independent of the class taxonomy.

### 4. What does `ConceptDataset` change relative to raw Fashion-MNIST?

It wraps the base dataset so `__getitem__` returns `(x, y, c)` instead of `(x, y)`, where `c` comes from `get_concepts(y)`.

### 5. Why is the backbone shared across all models?

So the comparison focuses on the effect of the output structure and bottleneck design, not on unrelated changes in feature extractor capacity.

## Architecture and tensor shapes

### 6. What is the shape after each stage of `CNNBackbone`?

`[B, 1, 28, 28] -> [B, 32, 28, 28] -> [B, 32, 14, 14] -> [B, 64, 14, 14] -> [B, 64, 7, 7] -> [B, 3136] -> [B, 128]`.

### 7. Why does `CrossEntropyLoss` not need softmax inside the model?

Because `CrossEntropyLoss` expects raw logits and internally applies the appropriate log-softmax computation.

### 8. Why is sigmoid correct for concepts?

Because the 8 concepts are multi-label, not mutually exclusive. Several can be true at once.

### 9. Does the strict CBM use hard binary concepts?

No. In this implementation, `CBM.forward` feeds predicted concept probabilities from `ConceptModel` directly into the final linear classifier.

### 10. Why does that detail matter?

Because the bottleneck is still concept-based, but it is softer than thresholding to hard 0/1 concept values. The classifier can use uncertainty information.

## Training-loop precision

### 11. Walk through one baseline training step exactly.

For batch `(x, y, _)`: move tensors to device, zero gradients, compute logits `[B, 10]`, compute cross-entropy loss against labels `[B]`, backpropagate, step Adam, and add `loss.item()` to the epoch total.

### 12. What does the printed epoch loss actually mean?

It is the sum of per-batch average losses across the epoch, not a dataset-normalized mean loss.

### 13. Is `model.train()` called everywhere it should be?

No. It is called in `train_baseline`, but not explicitly in `train_concept`, the CBM training block, or the hybrid training loop. Those models are still in train mode by default after initialization, so the code runs correctly, but the training mode handling is less explicit than ideal.

### 14. Why is that especially relevant for the hybrid?

Because `HybridCBM` contains `nn.Dropout`. If the model were accidentally left in eval mode, the side-channel dropout would be disabled and the experiment would change.

### 15. How is the hybrid objective defined?

As `CrossEntropyLoss(y_pred, y) + BCELoss(c_pred, c)` in cell 23, with equal coefficient 1 on both terms.

### 16. What is missing from that objective design?

No tuning of the relative weighting between class accuracy and concept prediction.

## Evaluation and methodology

### 17. How is AUROC computed here?

`evaluate_full` applies softmax to class logits, stores all class probabilities and labels, then calls `roc_auc_score(..., multi_class='ovr')`.

### 18. Why might accuracy and AUROC tell slightly different stories?

Accuracy depends only on the top predicted class, while AUROC measures ranking quality across thresholds and classes.

### 19. Why is concept F1 evaluated after thresholding at 0.5?

Because F1 is defined on binary predictions. The code converts concept probabilities to 0/1 decisions using a fixed threshold.

### 20. Do we have a validation split?

No. The notebook uses only train and test loaders.

### 21. What is the consequence of having no validation split?

Hyperparameter selection and model selection are weaker. In particular, the dropout sweep is compared on the test set instead of a separate validation set.

### 22. Which hybrid model is actually evaluated for concept F1 in cell 26?

The last one created in the dropout loop, which is the model with `dropout_p=0.9`.

### 23. Which hybrid model is used in the intervention experiment?

Also the last one bound to the variable `model` after the dropout loop, so again the `p=0.9` model.

### 24. Why is that a problem?

Because the notebook does not separately preserve the best hybrid model before running concept and intervention analysis.

## Harder conceptual questions

### 25. Why does the strict CBM lose accuracy even though concept F1 is high?

High concept F1 means the concepts are predicted well, not that the 8-concept representation is sufficient to distinguish all 10 classes. The bottleneck can still discard class-relevant detail.

### 26. Give an example of what "concept insufficiency" means in this repo.

Different classes can share overlapping or similar concept patterns. If two classes are not cleanly separable from the 8-dimensional concept vector alone, the strict CBM cannot recover the missing distinction.

### 27. Why might the hybrid outperform the strict CBM even if concept predictions are almost equally good?

Because the side channel carries residual image information that is not present in the concept vector.

### 28. Is the intervention code a clean estimate of causal concept importance?

No. For the hybrid model, original predictions use both concept and side branches, but the post-flip recomputation uses only the concept branch.

### 29. What would be a cleaner intervention implementation?

For the hybrid, keep the side-channel term fixed and replace only the concept contribution, for example compare `y_c + y_s` with `y_c_flipped + y_s`.

### 30. Why is `BCEWithLogitsLoss` a technically better default here?

Because it combines sigmoid and binary cross-entropy in a numerically more stable form than applying sigmoid first and then `BCELoss`.

### 31. What part of the repo most directly supports the claim that concepts are learnable?

The per-concept F1 scores from cell 17 and the macro F1 from `evaluate_concepts`.

### 32. What part most directly supports the claim that strict bottlenecks hurt label performance?

The baseline versus CBM comparison in cell 13 versus cell 20, and the summary in `report.md`.

### 33. What part most directly supports the claim that the hybrid recovers performance?

The dropout sweep results collected in cell 23 and summarized in `report.md`.

### 34. Can you prove from this repo that the hybrid is more interpretable than the baseline?

Not in a strong formal sense. What the repo shows is that the hybrid has an explicit concept interface and that concept manipulations change predictions. That supports greater inspectability and steerability, but not a full formal proof of interpretability.

### 35. What would you change first if you had one extra day?

Add a validation split, save the best hybrid checkpoint, and fix the intervention implementation so only the concept term changes during recomputation.

## Red-flag questions

These are the questions a professor may ask if they think the team only memorized the story and not the code.

### 36. After cell 7, what exactly is inside each batch returned by `train_loader`?

Three tensors: image batch `x`, label batch `y`, and concept batch `c`.

### 37. If `evaluate_full` were called on the raw loader from cell 3, would it work unchanged?

No. `evaluate_full` expects each batch to unpack as `(x, y, _)`, so it assumes the concept-augmented loader defined after cell 7.

### 38. Which parameters are updated when training the CBM?

Only `cbm.classifier.parameters()` are passed to Adam in cell 19.

### 39. Why is `CBM.forward` wrapped in `torch.no_grad()` around the concept model?

To freeze the concept predictor during CBM classifier training.

### 40. What is the exact shape of the tensor entering `self.fc` in `CNNBackbone`?

`[B, 3136]`, because the pooled feature maps are `[B, 64, 7, 7]`.

### 41. What is the exact shape of `c_pred` in the hybrid loop?

`[B, 8]`.

### 42. What is the exact shape of `y_pred` in the hybrid loop?

`[B, 10]`.

### 43. Why is dropout only applied to the side branch and not to the concept branch?

Because the experiment is specifically testing whether weakening the shortcut path changes the tradeoff. That design choice is visible in `HybridCBM.forward`.

### 44. Is the baseline directly comparable to the hybrid in training budget?

Not perfectly. The baseline trains for 5 epochs, while each hybrid variant trains for 3 epochs. So the comparison is informative, but not perfectly controlled for training duration.

### 45. Why is it dangerous to say "the hybrid uses concepts more at higher dropout"?

Because the repo does not directly measure concept reliance by dropout value. It only measures final class metrics across dropout settings, plus concept evaluation and intervention on the last hybrid instance.

# 9. "If You Get Stuck" Answers

- "The exact number depends on whether we quote the notebook outputs or `report.md`, and the repo does not explain that mismatch. The qualitative conclusion is the same."
- "The code shows this clearly in `project1.ipynb`; the strict CBM freezes the concept model and trains only the final classifier."
- "A limitation here is that the concepts are generated from labels, so they are not independent human annotations."
- "The tradeoff is accuracy versus a cleaner concept interface."
- "The hybrid is better described as more steerable than the baseline, not fully interpretable."
- "A stronger version of this project would keep a validation split and save the best model checkpoint."
- "The intervention result is suggestive, but not methodologically clean in the current implementation."
- "With more time, we would retune the hybrid loss weighting and fix the intervention code."
- "The code supports that the concepts are useful; it does not prove they are sufficient."
- "Unclear from repo, but the likely explanation is that the report and notebook were produced from different runs."

# 10. 2-Minute Presentation Script

This repository studies concept bottleneck models on Fashion-MNIST. In `project1.ipynb`, the base dataset is loaded with torchvision, then wrapped by `ConceptDataset` so that each training example contains not just an image and class label, but also an 8-dimensional concept vector produced by `get_concepts`. Those concepts are hand-designed binary attributes such as footwear, sleeves, collar, and long garment, and they are generated directly from the class label.

All four models use the same `CNNBackbone`, which maps an input image of shape `[B, 1, 28, 28]` to a hidden representation of shape `[B, 128]`. The `BaselineModel` sends that directly to 10 class logits. The `ConceptModel` predicts 8 concept probabilities. The strict `CBM` freezes the concept model and trains a classifier only on the 8 predicted concepts. The `HybridCBM` keeps the concept path but also adds a direct side-channel classifier from the hidden features, and its final class logits are the sum of the concept-based logits and the side-channel logits.

The results show the expected tradeoff. The baseline CNN has the best label accuracy. The strict concept bottleneck loses clear performance, which suggests that the 8 concepts do not fully preserve all class-relevant information. The hybrid model recovers most of that lost accuracy, according to the dropout sweep summarized in `report.md`. Concept F1 scores are high overall, so the concept layer is learnable, and the intervention code shows that changing concept values changes predictions. At the same time, the repo has limitations: there is no validation split, results differ slightly between notebook outputs and the report, and the intervention implementation for the hybrid is not perfectly clean.

# 11. Individual Teammate Study Guide

## A. Data / preprocessing person

Must understand:

- how Fashion-MNIST is loaded in cell 3
- what `transforms.ToTensor()` does
- that there is no augmentation and no explicit normalization
- how `get_concepts(label)` maps classes to 8 binary concepts
- why concept labels are deterministic and what that implies
- how `ConceptDataset` changes sample structure and why later code depends on that

## B. Architecture person

Must understand:

- exact `CNNBackbone` layer order
- exact tensor shapes through the backbone
- how `BaselineModel`, `ConceptModel`, `CBM`, and `HybridCBM` differ
- the difference between soft concept probabilities and hard thresholded concepts
- why the side channel exists in `HybridCBM`

## C. Training / optimization person

Must understand:

- the baseline loop in `train_baseline`
- the concept loop in `train_concept`
- that `model.train()` is explicit only in the baseline function
- how CBM freezing is implemented
- why only `cbm.classifier.parameters()` are optimized
- the exact hybrid loss formula and its equal weighting

## D. Evaluation / results person

Must understand:

- how `evaluate_full` computes accuracy and AUROC
- how concept F1 is computed
- why notebook metrics and `report.md` metrics differ slightly
- what is safe and unsafe to conclude from the results
- which hybrid instance is actually used for final concept/intervention evaluation

## E. Limitations / defense person

Must understand:

- no validation split
- no seed control
- no saved checkpoints
- label-derived concepts
- non-clean hybrid intervention implementation
- why the repo supports a narrow claim, not a universal one

## Minimum standard for everyone

Everyone should be able to answer these five questions without hesitation:

- Where are the concepts defined, and what are they?
- How does `CBM` differ from `HybridCBM` in actual code?
- What tensors flow through the backbone and heads?
- Why does the strict bottleneck lose accuracy?
- What are the main methodological weaknesses of this repo?
