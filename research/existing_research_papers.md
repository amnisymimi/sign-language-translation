## Existing Research Papers

### 1. Korean Sign Language Recognition and Sentence Generation (2026)

Jeong, S.-Y., Jeong, H.-Y., & Ihm, S.-Y. (2026). Korean sign language recognition and sentence generation through data augmentation. *Computers, Materials & Continua, 87*(2), 87. https://doi.org/10.32604/cmc.2026.074016

- **Focus:** Uses MediaPipe, OpenCV, and LSTM to recognize hand gestures and reconstruct sentences.
- **Findings:** Achieved 81.25% sign recognition accuracy and 95% sentence generation accuracy.
- **Relevance:** Provides a baseline for hand landmark extraction and temporal recognition.
- **Our improvement:** Incorporates facial expressions and other non-manual features.

### 2. The Importance of Facial Features in Vision-based Sign Language Recognition (2025)

Pham, D. N., & Avramidis, E. (2025). The importance of facial features in vision-based sign language recognition: Eyes, mouth or full face? In *Adjunct Proceedings of the 25th ACM International Conference on Intelligent Virtual Agents* (pp. 1–7). ACM. https://doi.org/10.1145/3742886.3756718

- **Focus:** Compares eyes, mouth, and full-face features using CSN and MViT.
- **Findings:** Finds that mouth features improve recognition when combined with body movements.
- **Relevance:** Supports integrating facial features with manual signing features.

### 3. Automatic Gloss-level Data Augmentation for Sign Language Translation (2022)

Jang, J. Y., Park, H.-M., Shin, S., Shin, S., Yoon, B., & Gweon, G. (2022). Automatic gloss-level data augmentation for sign language translation. In *Proceedings of the 13th Language Resources and Evaluation Conference* (pp. 6808–6813). European Language Resources Association. https://aclanthology.org/2022.lrec-1.734/

- **Focus:** Uses paraphrasing, synonym replacement, and blank replacement to expand KSL training data.
- **Findings:** Reports improved translation performance.
- **Relevance:** Supports data augmentation strategies to address limited KSL datasets.

### 4. KoSign: Introducing the NIASL2021 Dataset (2022)

Huerta-Enochian, M., Lee, D. H., Myung, H. J., Byun, K. S., & Lee, J. W. (2022). KoSign sign language translation project: Introducing the NIASL2021 dataset. In *Proceedings of the 7th International Workshop on Sign Language Translation and Avatar Technology: The Junction of the Visual and the Textual: Challenges and Perspectives* (pp. 59–66). European Language Resources Association. https://aclanthology.org/2022.sltat-1.9/

- **Focus:** Introduces a KSL dataset containing videos, hand keypoints, gloss annotations, and eight types of non-manual signals.
- **Relevance:** Provides a potential dataset for investigating manual and non-manual feature integration.

### 5. Cross-modality Data Augmentation for End-to-End Sign Language Translation (2023)

Ye, J., Jiao, W., Wang, X., Tu, Z., & Xiong, H. (2023). Cross-modality data augmentation for end-to-end sign language translation. In *Findings of the Association for Computational Linguistics: EMNLP 2023* (pp. 13558–13571). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.findings-emnlp.904

- **Focus:** Introduces cross-modality mix-up and knowledge distillation to connect visual sign features with gloss and text representations.
- **Findings:** Improves sign-to-text translation.
- **Relevance:** Supports multimodal feature fusion and sentence generation approaches.

### 6. ADAT: Time-Series-Aware Adaptive Transformer for Sign Language Translation (2026)

Shahin, N., & Ismail, L. (2026). ADAT novel time-series-aware adaptive transformer architecture for sign language translation. *Scientific Reports, 16*, Article 6551. https://doi.org/10.1038/s41598-026-36293-9

- **Focus:** Uses convolution, sparse attention, and adaptive gating to capture short- and long-term temporal dependencies.
- **Relevance:** Supports temporal modeling to understand hand movements and facial expressions over time.

---

## Overall Research Comparison

| Paper | Main Contribution | Relevance to Our Project |
|---|---|---|
| Paper 1 | Hand gesture recognition | Hand recognition baseline |
| Paper 2 | Facial feature integration | Non-manual feature extraction |
| Paper 3 | Gloss-level data augmentation | KSL data augmentation |
| Paper 4 | NIASL2021 KSL dataset | Dataset and non-manual annotations |
| Paper 5 | Cross-modal data augmentation | Multimodal feature fusion |
| Paper 6 | Temporal Transformer architecture | Temporal modeling |

## Our Research Gap

Previous studies have explored sign recognition, facial features, data augmentation, and translation through different approaches and combinations.

Our project aims to integrate these aspects into a unified Korean Sign Language translation framework, focusing on three key areas:

1. **Multimodal Features:** Combining manual features (hand movements and gestures) with non-manual features (facial expressions, eye gaze, and head movements).

2. **Temporal Alignment:** Understanding the temporal relationships between manual and non-manual features throughout a signing sequence.

3. **Context-Aware Translation:** Generating meaningful and grammatically natural Korean sentences by incorporating information from both manual and non-manual features.

### Main Research Contribution

Moving beyond hand gesture recognition towards **multimodal, temporally aligned Korean Sign Language translation**, with particular emphasis on the grammatical information conveyed through non-manual markers.
