# Concept Bottleneck Models on Fashion-MNIST

## 1. Overview

This project studies whether concept bottleneck models (CBMs) can make Fashion-MNIST classification more interpretable without giving up too much predictive performance. The repository implements four components:

- a baseline CNN classifier, `x -> y`
- a concept predictor, `x -> c`
- an independent CBM, `x -> c -> y`
- a hybrid CBM with both a concept path and a direct image side channel

The core question is the standard CBM trade-off: explicit concepts improve transparency and allow interventions, but they can also reduce accuracy if the concept set is too coarse to fully explain the class label.

## 2. Setup

The dataset is Fashion-MNIST: `28 x 28` grayscale images from 10 clothing classes. All models use the same CNN backbone with two convolutional blocks, max pooling, and a fully connected layer that produces a 128-dimensional hidden representation. The baseline and concept predictor are trained for 5 epochs; hybrid variants are trained for 3 epochs each.

Concept labels are generated deterministically from the class label. The 8 binary concepts are:

- footwear
- closed footwear
- footwear or bag
- sleeves
- collar
- long garment
- outerwear layer
- legwear or footwear

This makes supervision clean and reproducible, but it also limits semantic richness. The concepts capture broad structure well, yet they are not expressive enough to fully reconstruct every 10-way class distinction.

Evaluation uses test accuracy and multiclass AUROC for label prediction, and per-concept plus macro F1 for concept prediction.

## 3. Models

The baseline CNN is a direct classifier from image features to 10 class logits. It serves as the performance reference.

The concept predictor uses the same backbone but outputs 8 sigmoid probabilities and is trained with binary cross-entropy.

The CBM is an **independent** bottleneck model. First, the concept predictor is trained. Then its concept outputs are frozen and a linear classifier is trained on top of them. This gives maximum interpretability because label prediction can only use the concept vector, but it also makes the model sensitive to concept insufficiency.

The hybrid CBM combines a concept path and a residual shortcut. If `h` is the hidden representation, the model predicts concepts `c = sigmoid(W_c h)`, computes concept logits `W_y c`, adds a side-channel label prediction `W_s dropout(h)`, and outputs the sum. It is trained jointly with label loss plus concept loss. A dropout sweep on the side channel tests whether weakening the shortcut encourages more reliance on concepts.

## 4. Results

Table 1 summarizes the main predictive results.

| Model | Accuracy | AUROC |
|---|---:|---:|
| Baseline CNN | 0.9080 | 0.9938 |
| Independent CBM | 0.8109 | 0.9824 |
| Hybrid CBM, `p=0.0` | 0.8981 | 0.9931 |
| Hybrid CBM, `p=0.1` | 0.8947 | 0.9926 |
| Hybrid CBM, `p=0.3` | 0.9035 | 0.9931 |
| Hybrid CBM, `p=0.5` | 0.9001 | 0.9926 |
| Hybrid CBM, `p=0.7` | 0.8959 | 0.9927 |
| Hybrid CBM, `p=0.9` | 0.8989 | 0.9917 |

Three patterns matter most.

First, the baseline remains the strongest classifier. Second, the strict CBM loses substantial performance, dropping from `0.9080` to `0.8109` accuracy. This is the clearest evidence of the interpretability-performance trade-off in the notebook. Third, the hybrid model recovers most of that gap: every dropout setting stays near baseline, and the best hybrid (`p=0.3`) reaches `0.9035` accuracy, only `0.0045` below the baseline. In other words, the hybrid recovers most of the performance lost by the pure bottleneck while retaining an explicit concept interface.

Concept prediction quality is also strong. The standalone concept model reaches macro F1 `0.9513`, while the final hybrid instance reaches macro F1 `0.9431`. Several concepts are nearly perfect, especially footwear-related ones and `legwear or footwear`. The weaker concepts are `has collar` (`0.8404` standalone, `0.8237` hybrid), `is long garment` (`0.8857`, `0.8647`), and `is outerwear layer` (`0.9074`, `0.8859`). These weaker concept dimensions are plausible contributors to the remaining label gap between CBM-style models and the baseline.

The dropout sweep does not show a simple monotonic pattern. Accuracy peaks at `p=0.3`, while the full range from `p=0.0` to `p=0.9` remains fairly tight. That suggests the side channel helps, but the hybrid does not collapse when the shortcut is regularized aggressively.

## 5. Steerability

The repository also includes a concept intervention experiment on the final stored hybrid model. One concept is flipped at a time and the downstream prediction is recomputed. Two quantities are reported: average absolute probability change and label flip rate.

All 8 concepts produce substantial downstream effects. Average probability changes are tightly clustered around `0.13` to `0.14`, and label flip rates range from `0.3901` to `0.7975`. The strongest interventions by average probability change are `is long garment`, `has collar`, and `has sleeves`. The highest flip rates appear for `is closed footwear` and `is long garment`.

This shows that the concept pathway is not decorative: changing concepts materially changes the prediction. That makes the hybrid model more steerable than the baseline CNN, which has no explicit concept interface. However, the intervention study is still limited. It evaluates only the final hybrid instance, not all dropout settings, and it uses synthetic concept flips rather than verified semantic counterfactual edits.

## 6. Discussion and Conclusion

The main conclusion is straightforward. On Fashion-MNIST, concept supervision clearly improves transparency and intervention access, but a strict bottleneck costs accuracy. The independent CBM is the most interpretable model, yet its performance drop shows that the 8 concepts do not fully capture all label-relevant information. The hybrid architecture is the best practical compromise in this repository: it preserves explicit concepts, keeps concept F1 high, supports meaningful interventions, and stays very close to baseline classification performance.

The main limitations are also clear. The concept set is hand-coded and fairly coarse, the workflow is centered on a notebook rather than reproducible scripts and saved artifacts, and the intervention analysis is narrow. Even so, the experimental evidence is strong enough to support the core claim: hybrid CBMs can retain much of the predictive power of a standard CNN while offering a substantially more interpretable and steerable intermediate representation.
