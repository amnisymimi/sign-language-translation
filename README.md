# sign-language-translation
[2026-02] Deep Learning Team Project - Sign Language Translation

# Person 3 — Translation Model Development

**Project:** Sign Language Translation with Manual and Non-Manual Features  
**Branch:** `syamimi-dev`  
**Role:** Model Architecture, Training & Evaluation  
**Status:** In Development

## 1. Overview

This document describes Person 3's responsibilities in developing the translation component of our multimodal Korean Sign Language Translation system.

The primary goal is to design, implement, and evaluate a model that translates sign language features into Korean sentences.

The model will initially use manual features only, then incorporate non-manual features through multimodal fusion.

### Research Objective

Investigate whether incorporating facial expressions and other non-manual grammatical markers improves sign language translation compared with a manual-only baseline.

Three configurations are planned:

- **Model A:** Manual-only translation baseline
- **Model B:** Manual and non-manual features with simple fusion
- **Model C:** Manual and non-manual features with temporal cross-attention

These models will be evaluated using consistent data splits and evaluation metrics.

---

## 2. Responsibilities

| Week | Main Task | Expected Deliverable |
|---|---|---|
| Week 1 | Literature Review + Baseline Design | Research summary and baseline architecture |
| Week 2 | Hand-Only Baseline | Initial working baseline model |
| Week 3 | Translation Decoder | Working Korean text decoder |
| Week 4 | Multimodal Integration | End-to-end multimodal model |
| Week 5 | Final Training + Evaluation | Model comparison and evaluation results |

## 3. Week 1 — Literature Review + Baseline Design

### Tasks

- [ ] Review existing sign language recognition and translation research
- [ ] Compare CNN, RNN/LSTM, and Transformer architectures
- [ ] Study manual-only and multimodal translation methods
- [ ] Investigate encoder-decoder architectures
- [ ] Study attention and cross-attention mechanisms
- [ ] Identify suitable evaluation metrics
- [ ] Propose the hand-only baseline architecture
- [ ] Agree on model interfaces with other teammates

### Deliverables

- Literature review summary
- Comparison of candidate architectures
- Baseline model architecture diagram
- Proposed training and evaluation strategy

### Architecture Considerations

Potential approaches include:

**CNN:** Spatial feature extraction from video frames.

**RNN/LSTM:** Temporal modeling of sequential sign features.

**Transformer:** Attention-based modeling of temporal relationships.

A Transformer-based encoder-decoder is one candidate architecture, but the final choice will depend on data availability, computational resources, and preliminary experiments.

---

## 4. Week 2 — Hand-Only Baseline

### Tasks

- [ ] Implement baseline model architecture
- [ ] Define manual feature input dimensions
- [ ] Implement temporal sequence encoder
- [ ] Implement training and validation loops
- [ ] Define translation loss function
- [ ] Test the model with dummy feature sequences
- [ ] Train on a small real-data subset when available
- [ ] Record initial training results

### Proposed Baseline Architecture

```text
Sign Language Video
         |
  Hand Keypoints
         |
  Manual Features
         |
  Temporal Encoder
         |
 Translation Decoder
         |
  Korean Sentence
```

### Expected Input

```text
manual_features: [B, T, D_manual]
frame_mask: [B, T]
```

### Expected Output

```text
token_logits: [B, L, V]
```

Where:

- `B`: Batch size
- `T`: Number of video frames
- `D_manual`: Manual feature dimension
- `L`: Target token sequence length
- `V`: Vocabulary size

### Deliverable

A working hand-only translation baseline with reproducible training code.

---

## 5. Week 3 — Translation Decoder

### Tasks

- [ ] Select Korean text tokenization method
- [ ] Build or load a tokenizer
- [ ] Define vocabulary and special tokens
- [ ] Implement decoder architecture
- [ ] Implement causal attention masking
- [ ] Implement cross-attention to encoded sign features
- [ ] Implement token prediction and training loss
- [ ] Implement inference and sentence generation
- [ ] Test decoder using dummy encoder outputs
- [ ] Integrate decoder with the manual encoder

### Proposed Decoder Architecture

```text
Encoded Sign Features
         |
         v
   Cross-Attention <--- Korean Target Tokens
         |
   Decoder Layers
         |
   Linear Projection
         |
  Vocabulary Logits
         |
   Token Generation
         |
   Korean Sentence
```

The decoder will be trained using appropriate target token sequences and a token-level loss function, such as cross-entropy.

### Deliverable

A translation decoder capable of generating Korean text from encoded sign features.

---

## 6. Week 4 — Multimodal Model Integration

### Tasks

- [ ] Receive manual feature encoder outputs from Person 1's pipeline
- [ ] Receive facial feature outputs and fusion module from Person 2
- [ ] Verify temporal alignment and tensor dimensions
- [ ] Connect manual and non-manual feature streams
- [ ] Integrate the fusion module with the translation decoder
- [ ] Implement the full multimodal forward pass
- [ ] Verify masking and variable-length sequence handling
- [ ] Test training and inference on a small batch
- [ ] Debug integration issues with teammates

### Proposed Multimodal Architecture

```text
           Sign Language Video
                    |
            Feature Extraction
                    |
          +---------+---------+
          |                   |
   Manual Features     Facial Features
          |                   |
    Manual Encoder      Facial Encoder
          |                   |
          +---------+---------+
                    |
             Temporal Fusion
                    |
             Cross-Attention
                    |
            Translation Decoder
                    |
              Korean Sentence
```

### Deliverable

A working multimodal translation model that accepts both manual and non-manual features.

---

## 7. Week 5 — Final Training & Evaluation

### Tasks

- [ ] Finalize model configurations
- [ ] Train the manual-only baseline
- [ ] Train the simple-fusion multimodal model
- [ ] Train the cross-attention multimodal model if feasible
- [ ] Evaluate all models using the same held-out test split
- [ ] Calculate BLEU and chrF
- [ ] Compare translation performance
- [ ] Analyze successful and unsuccessful predictions
- [ ] Record hyperparameters and training settings
- [ ] Prepare final result tables and figures

### Planned Model Comparison

| Model | BLEU | chrF | Status |
|---|---|---|---|
| Manual-only baseline | TBD | TBD | Planned |
| Simple multimodal fusion | TBD | TBD | Planned |
| Temporal cross-attention | TBD | TBD | Planned |

No results will be reported until experiments are completed.

---

## 8. Collaboration with Other Members

### Person 1 — Dataset and Manual Features

Person 1 provides:

- Processed dataset metadata
- Manual keypoint sequences
- Training, validation, and test splits
- Sequence masks and feature documentation

### Person 2 — Facial Features and Fusion

Person 2 provides:

- Non-manual feature sequences
- Temporal alignment module
- Cross-attention fusion module
- Facial feature interface documentation

### Person 3 — Translation Model

Person 3 provides:

- Baseline architecture
- Translation encoder/decoder implementation
- Training and inference scripts
- Multimodal model integration
- Final evaluation results

All members should agree on input/output formats before integration.

---

## 9. Proposed Development Structure

```text
models/
    encoders/
    fusion/
    decoder/

training/
    train_baseline.py
    train_multimodal.py

evaluation/
    evaluate.py

configs/
    baseline.yaml
    multimodal.yaml

docs/
    person3_README.md
    literature_review.md
    architecture.md
```

These are proposed files and directories, not necessarily completed implementations.

---

## 10. Development Environment

**Language:** Python  
**Deep Learning Framework:** PyTorch  
**Development:** VS Code and GitHub  
**Training:** AWS EC2 GPU environment

Dependencies and setup instructions will be documented once the model implementation is finalized.

---

## 11. GitHub Workflow

Development branch: `syamimi-dev`

Before making changes:

```bash
git switch syamimi-dev
git pull origin syamimi-dev
```

After completing a task:

```bash
git add .
git commit -m "Implement translation model component"
git push origin syamimi-dev
```

Only stage files intended for the repository. Large datasets, model checkpoints, and credentials must remain outside Git.

Completed components should be reviewed through Pull Requests before merging into `main`.

---

## 12. Progress Tracking

### Week 1
- [ ] Literature review completed
- [ ] Baseline architecture proposed
- [ ] Model interface agreed with team

### Week 2
- [ ] Hand-only baseline implemented
- [ ] Initial training test completed

### Week 3
- [ ] Translation decoder implemented
- [ ] Baseline inference tested

### Week 4
- [ ] Multimodal components integrated
- [ ] End-to-end training tested

### Week 5
- [ ] Final training completed
- [ ] Model comparison completed
- [ ] Evaluation results documented

---

## 13. Success Criteria

The Person 3 component will be considered complete when:

1. The manual-only baseline can be trained and evaluated.
2. The translation decoder can generate Korean sentence predictions.
3. Manual and non-manual features can be integrated into the translation model.
4. The model supports reproducible training and inference.
5. Translation results can be compared across the planned configurations.
6. Code, model settings, and experimental results are documented.

**Current status:** Development planning and initial research.
