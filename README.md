# sign-language-translation
[2026-02] Deep Learning Team Project - Sign Language Translation

# Sign Language Translation with Manual and Non-Manual Features

**Deep Learning Team Project | Fall 2026**

A multimodal sign language translation project investigating how facial expressions and other non-manual grammatical markers can improve translation when combined with manual signing features.

## 1. Project Overview

Sign language communication involves more than hand gestures. Facial expressions, eyebrow movements, mouth shapes, and head movements can also convey important grammatical information.

This project aims to develop a deep learning-based sign language translation system that integrates **manual features** (hand and body movements) and **non-manual features** (facial expressions and head movements).

Our main research focus is understanding whether modeling the temporal relationship between manual and non-manual features can improve sign language translation.

### Research Questions

1. Does incorporating non-manual features improve translation performance compared with a manual-only baseline?
2. Can temporal alignment and cross-attention better capture relationships between manual and non-manual signals?
3. Which non-manual features contribute most to translation performance?

## 2. Proposed Architecture

The proposed system consists of two feature streams:

**Manual Stream**
- Hand keypoints
- Body/pose keypoints where appropriate
- Temporal encoding of manual movements

**Non-Manual Stream**
- Facial landmarks
- Eyebrow and mouth movements
- Head pose and movement
- Temporal encoding of facial information

The two streams are combined through a multimodal fusion module and passed to a translation decoder.

```text
                Sign Language Video
                         |
               Feature Extraction
                         |
              +----------+----------+
              |                     |
       Manual Features      Non-Manual Features
       (Hand / Body)        (Face / Head)
              |                     |
       Manual Encoder       Facial Encoder
              |                     |
              +----------+----------+
                         |
             Temporal Alignment
                         |
              Cross-Attention
                         |
              Translation Decoder
                         |
                Korean Sentence
```

The architecture is provisional and may change following baseline experiments.

## 3. Dataset

### Primary Dataset

**NIKL Sign Language Parallel Corpus (2022)**

The project investigates the use of Korean Sign Language video data and corresponding linguistic annotations.

Dataset preparation includes:

- Investigating video and annotation formats
- Matching videos with corresponding labels
- Extracting manual and non-manual features
- Aligning features temporally
- Creating training, validation, and test datasets
- Checking data quality and annotation consistency

Additional datasets may be considered if their languages, annotations, and licensing conditions are suitable.

**Note:** Raw datasets, large video files, and extracted features are not stored in this repository.

## 4. Experimental Plan

We plan to compare three model configurations.

| Model | Description | Purpose |
|---|---|---|
| A: Manual-only | Manual features with translation model | Establish baseline |
| B: Simple fusion | Manual and non-manual features combined | Measure facial feature contribution |
| C: Temporal cross-attention | Two-stream architecture with temporal fusion | Evaluate temporal relationships |

### Evaluation

Potential evaluation methods include:

- BLEU and chrF for translation quality
- Qualitative translation examples
- Ablation experiments on non-manual features
- Comparison of manual-only and multimodal models
- Analysis of temporal alignment and attention behavior

Evaluation methods will be finalized based on dataset suitability and project requirements.

## 5. Project Timeline

| Week | Person 1: Data Pipeline | Person 2: Facial/Fusion | Person 3: Translation Model |
|---|---|---|---|
| 1 | Dataset investigation | Research facial features | Literature review and baseline design |
| 2 | Dataset preparation and hand keypoints | Facial feature extraction | Hand-only baseline |
| 3 | Complete manual feature pipeline | Temporal alignment module | Translation decoder |
| 4 | Prepare and validate integrated dataset | Complete cross-attention module | Integrate both streams |
| 5 | Dataset quality analysis | Alignment visualization and ablation support | Final training and evaluation |

### Shared Responsibilities

All team members participate in:

- Defining data and model interfaces
- Code review and debugging
- Integration testing
- Experiment documentation
- Final report and presentation

The team aims to complete an initial end-to-end prototype by Week 3 and reserve sufficient time for integration, evaluation, and reporting.

## 6. Team Collaboration

The project uses GitHub for version control, task management, and collaborative development.

### Development Workflow

1. Create or select an assigned GitHub Issue.
2. Work on a separate development branch.
3. Commit changes with descriptive messages.
4. Push changes to GitHub.
5. Create a Pull Request to merge into `main`.
6. Review and test changes before merging.

### Branch Structure

| Branch | Purpose |
|---|---|
| `main` | Stable shared project |
| `inkyung-dev` | Person 1 development |
| `nuhaa-dev` | Person 2 development |
| `syamimi-dev` | Person 3 development |

Branch names for other members will be confirmed once everyone has joined.

## 7. Proposed Repository Structure

```text
sign-language-translation/
|
|-- preprocessing/
|   |-- README.md
|   |-- ...
|
|-- features/
|   |-- manual/
|   |-- nonmanual/
|
|-- models/
|   |-- encoders/
|   |-- fusion/
|   |-- decoder/
|
|-- training/
|
|-- evaluation/
|
|-- configs/
|
|-- docs/
|
|-- README.md
|-- .gitignore
|-- requirements.txt
```

This structure represents the planned organization. Directories and files will be added as development progresses.

## 8. Technology Stack

**Programming**
- Python

**Deep Learning**
- PyTorch

**Feature Extraction (Proposed)**
- MediaPipe
- OpenCV

**Modeling**
- CNN / RNN / Transformer-based approaches
- Attention and cross-attention mechanisms

**Development**
- Git and GitHub
- Visual Studio Code
- AWS EC2 GPU environment

The final tools and architectures will be selected after initial experimentation.

## 9. Project Status

**Status: In Development**

Current priorities:

- Dataset investigation and preprocessing
- Literature review and architecture design
- Facial feature research
- Shared data format specification
- Initial baseline implementation

Training results and evaluation metrics will be added once experiments have been completed.

## 10. Team

| Member | Primary Responsibility |
|---|---|
| Person 1 | Dataset preparation and manual feature pipeline |
| Person 2 | Facial features, temporal alignment, and multimodal fusion |
| Person 3 | Baseline architecture, translation decoder, and model training |

## 11. References

Research papers, dataset documentation, and implementation references will be added as the literature review progresses.

## 12. Acknowledgements

This project is developed as part of the Fall 2026 Deep Learning course.

The team acknowledges the dataset creators and research communities contributing to sign language recognition and translation research.
