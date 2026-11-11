# Facial Feature Research Summary and Proposed Extraction Method

## 1. Research Summary

Non-manual markers (NMMs) are visual components of sign language that communicate grammatical information through facial expressions, eye movements, mouth shapes, and head movements.

For our project, we focus on extracting facial information that may contribute to grammatical interpretation in Korean Sign Language (KSL).

The research and implementation focus on four main facial components:

| Facial Component | Features Investigated | Potential Linguistic Information |
|---|---|---|
| Eyebrows | Raised, lowered, and furrowed eyebrows | Questions, topic marking, focus, and emphasis |
| Eyes | Eye blinking, widening, and squinting | Focus and grammatical distinctions |
| Mouth | Mouth opening, lip funneling, puckering, and pressing | Additional linguistic information and mouth-related expressions |
| Head | Head orientation and movement | Questions, negation, and emphasis |

### Comparison of Landmark-Based and Image-Based Facial Representations

Landmark-based and image-based representations are two approaches for extracting facial features in sign language translation. 

Landmark-based methods: use facial landmark coordinates and blendshape scores to represent facial movements, such as eyebrow movements, eye expressions, and mouth shapes. They produce structured numerical features that are relatively lightweight, easier to interpret, and suitable for integration with manual features such as hand landmarks. However, their performance may be affected by occlusion, head rotation, or inaccurate landmark detection. 

Image-based methods: use facial images or cropped face regions as input to deep learning models, such as CNNs or Vision Transformers, which automatically learn visual representations. These methods can capture more detailed facial information and subtle expressions but generally require more computational resources and training data.

For this project, we selected a **landmark-based approach** using MediaPipe Face Landmarker because it provides structured facial landmarks and blendshape scores that can be temporally aligned with manual signing features, making it a practical starting point for multimodal KSL translation.

## 2. Technology and Approach

We selected **MediaPipe Face Landmarker** for facial feature extraction.

MediaPipe provides facial landmark coordinates, blendshape scores, and facial transformation matrices that can be used to represent facial movements and expressions across video frames.

Our implementation uses:

- **OpenCV:** Reads and processes video frames.
- **MediaPipe Face Landmarker:** Detects facial landmarks, blendshapes, and facial transformation matrices.
- **NumPy:** Stores and processes numerical feature sequences.
- **Pandas:** Organizes extracted facial features into structured data.
- **Matplotlib:** Visualizes facial feature changes over time.
- **SciPy:** Converts rotation matrices into head-pose angles.

The implementation is designed to support:
- Extraction of 478 facial landmark coordinates (when all landmarks are detected).
- Extraction of selected facial blendshape scores.
- Tracking of facial feature changes across consecutive frames.
- Estimation of head orientation using pitch, yaw, and roll.
- Visualization of facial and head movement over time.

## 3. Proposed Facial Feature Extraction Method

Our proposed extraction pipeline consists of the following stages:

### Step 1: Video Input and Frame Extraction

- Load the KSL signing video using OpenCV. 
- Retrieve video information, including FPS, total frames, and duration.
- Process consecutive frames for facial feature extraction.
- Convert frames from BGR to RGB format for MediaPipe processing.

### Step 2: Facial Landmark Detection

- Initialize MediaPipe Face Landmarker.
- Process each frame using video mode.
- Detect facial landmarks and extract their normalized x, y, and z coordinates.
- Record whether a face is detected in each frame.

### Step 3: Facial Blendshape Extraction

Extract selected blendshape scores to represent three facial components:

| Component | Selected Features |
|---|---|
| Eyebrows | `browInnerUp`, `browDownLeft`, `browDownRight`, `browOuterUpLeft`, `browOuterUpRight` |
| Eyes | `eyeBlinkLeft`, `eyeBlinkRight`, `eyeWideLeft`, `eyeWideRight`, `eyeSquintLeft`, `eyeSquintRight` |
| Mouth | `jawOpen`, `mouthFunnel`, `mouthPucker`, `mouthSmileLeft`, `mouthSmileRight`, `mouthPressLeft`, `mouthPressRight` |

A total of **18 selected blendshape features** are extracted for each video frame.

### Step 4: Head Pose and Movement Extraction

The updated implementation also uses MediaPipe facial transformation matrices to estimate head orientation.

Three head-pose features are extracted:

| Feature | Description |
|---|---|
| Pitch | Upward or downward head rotation |
| Yaw | Left or right head rotation |
| Roll | Sideways head tilt |

Temporal changes in these angles are also calculated to represent head movement over time.

This extends the facial representation beyond blendshape scores and provides additional information about head orientation and movement.

### Step 5: Temporal Feature Extraction

- Calculate frame-to-frame changes in blendshape scores.
- Calculate the rate of facial feature changes using timestamps.
- Calculate temporal changes in head-pose angles.
- Apply a five-frame rolling average to smooth eyebrow movement.
- Visualize facial feature changes over time.

This allows the model to capture dynamic facial movements rather than relying only on individual frames.

### Step 6: Missing-Value Handling

- Identify frames where facial detection fails.
- Interpolate short gaps of up to five consecutive frames.
- Fill remaining missing feature values with zero.
- Generate a face detection mask to identify frames without detected faces.
- Convert feature values into numerical arrays.

### Step 7: Facial Feature Sequence Generation

The extracted features are organized into sequential representations for future deep learning models.

| Output File | Description |
|---|---|
| `X_face.npy` | Facial blendshape and head-pose feature sequence |
| `face_mask.npy` | Face detection mask indicating whether a face was detected in each frame |
| `facial_features.csv` | Complete facial landmark and blendshape data |
| `selected_facial_features.csv` | Selected eyebrow, eye, and mouth features |
| `temporal_facial_features.csv` | Facial features with temporal changes |
| `complete_facial_features.csv` | Combined facial and head-pose features, including temporal head movement information |

The final feature dimensions depend on which features are successfully extracted and included in the sequence.

## 4. Proposed Extraction Pipeline

```text
             KSL Signing Video
                     |
                     v
           OpenCV Frame Extraction
                     |
                     v
          MediaPipe Face Landmarker
                     |
          +----------+----------+
          |          |          |
          v          v          v
   Facial Landmarks Blendshape  Facial Transformation
   (x, y, z)        Scores      Matrices
                   (18)             |
                     |              v
                     |        Head Pose Extraction
                     |        (Pitch, Yaw, Roll)
                     |              |
                     +------+-------+
                            |
                            v
                 Temporal Feature Extraction
                            |
                            v
                     Feature Smoothing
                            |
                            v
                   Missing-Value Handling
                            |
                            v
                  Facial Feature Sequences
                            |
                 +----------+----------+
                 |                     |
                 v                     v
              X_face.npy          face_mask.npy
                 |                     |
                 +----------+----------+
                            |
                            v
                  Multimodal Feature Fusion
                    (Future Integration)
```

## 5. Current Implementation Status

| Research / Implementation Task | Status |
|---|---|
| Research non-manual grammatical markers | Completed |
| Investigate eyebrow, mouth, and head movements | Completed |
| Research MediaPipe Face Mesh / Face Landmarker | Completed |
| Implement facial landmark extraction from videos | Implemented |
| Investigate landmark-based facial representation | Implemented |
| Compare landmark-based and image-based representations | Research completed; no experimental comparison yet |
| Identify methods for modeling facial features over time | Completed |
| Implement temporal facial feature extraction | Implemented |
| Propose facial feature extraction pipeline | Completed |
| Generate facial feature sequences | Implemented; execution validation pending |
| Add head-pose extraction (pitch, yaw, roll) | Added to updated notebook; validation pending |
| Add temporal head movement features | Added to updated notebook; validation pending |

## 6. This Week's Progress (November 1-4, 2026)

This week, we focused on preparing the project environment and extending the facial feature extraction pipeline.

### Repository and Environment Setup

- Completed the GitHub repository structure
- Connected GitHub to Google Colab.
- Set up the Colab notebook and Google Drive.
- Confirmed that the Colab environment provides a T4 GPU.

### Facial Feature Extraction Improvements

- Reviewed and organized the existing MediaPipe facial feature extraction pipeline.
- Extended the notebook to include head-pose extraction using facial transformation matrices.
- Added pitch, yaw, and roll as head orientation features.
- Added temporal head movement calculations to capture changes in head orientation over time.
- Updated the proposed facial feature sequence to include both blendshape and head-pose information.

## 7. Future Improvements

The current implementation provides a foundation for non-manual feature extraction. Several improvements are planned:

1. **Detection Validation:** Evaluate facial detection performance across multiple KSL videos and different signing conditions.
2. **Multimodal Alignment:** Align facial and hand feature sequences using video timestamps.
3. **Model Integration:** Combine facial feature sequences with manual features for multimodal sign language translation.

## 8. Expected Contribution

This implementation establishes a facial feature extraction pipeline that converts visual facial information into structured numerical sequences.

By capturing eyebrow movements, eye expressions, mouth shapes, head orientation, and temporal facial changes, the extracted features can provide additional information beyond manual hand gestures.

The resulting facial feature sequences will serve as the non-manual input branch of our proposed multimodal Korean Sign Language translation framework.
