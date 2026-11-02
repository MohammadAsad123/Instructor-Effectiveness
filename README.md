# Instructor Effectiveness Modeling

An end-to-end data science and machine learning framework designed to evaluate, quantify, and predict instructor effectiveness across course batches using multidimensional educational telemetry data.

---

## 📌 Executive Summary

Evaluating teaching effectiveness in online and hybrid educational platforms is critical for student retention, curriculum optimization, and faculty development. However, evaluating instructors solely on subjective surveys or simple pass rates introduces significant bias.

This project implements a transparent, multi-dimensional evaluation system:
1. **Aggregates batch-level observations** ($2{,}000$ course cohorts) into instructor-level performance profiles ($120$ unique instructors across $25$ distinct courses).
2. **Formulates an Instructor Effectiveness Score** combining three core pillars: **Learner Outcomes (33.33%)**, **Learner Engagement (33.33%)**, and **Learner Feedback (33.33%)**.
3. **Establishes Effectiveness Tiers** (`Low`, `Medium`, `High`) via empirical terciles.
4. **Trains a Machine Learning Classifier** (Random Forest) to predict effectiveness tiers using only raw aggregated performance features.
5. **Evaluates model performance rigorously**, achieving **79.17% Test Accuracy** and **88.33% Mean 5-Fold Stratified Cross-Validation Macro F1**, outperforming the baseline by **+45.84%**.
6. **Analyzes model robustness, feature attributions, confounding variables, and deployment ethics** for real-world EdTech decision-support systems.

---

## 📊 Dataset Overview & Exploratory Data Analysis (EDA)

The dataset contains $2{,}000$ batch-level records detailing learner behavior and outcomes across various courses and instructors.

### Dataset Schema & Structure
- **Total Records:** 2,000 batches
- **Unique Instructors:** 120 instructors (ranging from 7 to 31 batches per instructor)
- **Unique Courses:** 25 courses (`C_01` to `C_25`, averaging ~80 batches per course)
- **Data Hygiene:** 0 missing values, 0 duplicate records

| Metric Category | Feature Name | Description | Range in Data |
| :--- | :--- | :--- | :--- |
| **Identifiers** | `batch_id` | Unique identifier for each course batch | B_0001 to B_2000 |
| | `instructor_id` | Identifier of the assigned instructor | I_001 to I_120 |
| | `course_id` | Identifier of the course offering | C_01 to C_25 |
| **Learner Outcomes** | `completion_rate` | Proportion of enrolled learners completing the batch | 0.300 – 0.980 |
| | `dropout_rate` | Proportion of enrolled learners dropping out | 0.020 – 0.700 |
| | `avg_score_improvement` | Average score delta between pre- and post-assessments | 15.00 – 35.00 |
| | `avg_quiz_score` | Mean score obtained across batch quizzes | 65.00 – 90.00 |
| **Learner Engagement** | `avg_watch_time` | Average video content watch completion percentage | 0.500 – 0.950 |
| | `assignment_submission_rate` | Proportion of assigned tasks submitted | 0.500 – 0.950 |
| | `forum_activity_rate` | Proportion of learners active in discussion forums | 0.100 – 0.400 |
| **Learner Feedback** | `avg_feedback_score` | Average learner rating for the batch | 3.500 – 5.000 |
| | `feedback_response_rate` | Proportion of batch learners who submitted feedback | 0.400 – 0.950 |

### Key Exploratory Findings
- **Multicollinearity between Retention Metrics:** `completion_rate` and `dropout_rate` exhibit a near-perfect inverse correlation ($r = -0.953$ at the batch level, $r = -0.990$ at the instructor level).
- **Non-Complementary Reporting:** Although strongly correlated, `completion_rate + dropout_rate` does not strictly equal $1.0$ (mean sum = $0.9977$, range: $0.8287$ to $1.1604$). This indicates independent tracking in platform logs (e.g., learners who became inactive without formally withdrawing).
- **Varied Scales & Distributions:** Engagement metrics (such as `forum_activity_rate` peaking around $25\%$) operate on fundamentally different numerical distributions than feedback scores ($3.5$ to $5.0$). Direct summation is invalid without normalization.

---

## 📐 Instructor Effectiveness Scoring Framework

To evaluate instructors rather than individual batch anomalies, batch records are aggregated by `instructor_id` using mean values.

### 1. Normalization & Directional Alignment
Each feature $X$ is scaled via Min-Max normalization to the range $[0, 1]$:

$$X_{\text{norm}} = \frac{X - X_{\min}}{X_{\max} - X_{\min}}$$

Because higher dropout denotes poorer performance, its normalized value is directionally inverted:

$$\text{DropoutScore} = 1 - \text{Dropout}_{\text{norm}}$$

### 2. Multi-Dimensional Pillar Formulations

#### A. Learner Outcomes Dimension (Weight: 33.33%)
To prevent retention metrics from dominating the outcome score due to collinearity, `completion_rate` and `dropout_rate` are combined into an averaged **Retention Score**:

$$\text{RetentionScore} = \frac{\text{CompletionScore} + \text{DropoutScore}}{2}$$

$$\text{OutcomeScore} = \frac{\text{RetentionScore} + \text{ScoreImprovementScore} + \text{QuizScore}}{3}$$

#### B. Learner Engagement Dimension (Weight: 33.33%)
Captures content consumption, coursework adherence, and collaborative participation:

$$\text{EngagementScore} = \frac{\text{WatchTimeScore} + \text{AssignmentSubmissionScore} + \text{ForumActivityScore}}{3}$$

#### C. Learner Feedback Dimension (Weight: 33.33%)
Combines satisfaction rating with response coverage (addressing response bias):

$$\text{FeedbackScore} = \frac{\text{FeedbackRatingScore} + \text{FeedbackResponseScore}}{2}$$

### 3. Composite Effectiveness Score & Tier Classification
The final composite index equally weights all three pillars:

$$\text{EffectivenessScore} = \frac{\text{OutcomeScore} + \text{EngagementScore} + \text{FeedbackScore}}{3}$$

Instructors ($N=120$) are segmented into three equal terciles (`pd.qcut`, 40 instructors each):

| Effectiveness Tier | Quantile Cut | Score Range | Mean Composite Score | Outcome Mean | Engagement Mean | Feedback Mean |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Low** | Bottom 33.3% | $[0.177, 0.448]$ | **0.333** | 0.306 | 0.388 | 0.306 |
| **Medium** | Middle 33.3% | $(0.448, 0.584]$ | **0.504** | 0.502 | 0.521 | 0.489 |
| **High** | Top 33.3% | $(0.584, 0.954]$ | **0.692** | 0.698 | 0.675 | 0.705 |

---

## 🤖 Machine Learning Modeling

The tier assignment is framed as a supervised multi-class classification task:
- **Target Variable ($y$):** `effectiveness_tier` (`Low`, `Medium`, `High`).
- **Feature Set ($X$):** The 9 raw aggregated instructor-level metrics.
- **Leakage Prevention:** `instructor_id`, `course_id`, `num_batches`, dimensional sub-scores, and the composite effectiveness score are strictly excluded from predictors.
- **Data Partitioning:** Stratified 80/20 train-test split ($96$ training instructors, $24$ test instructors, preserving equal 8:8:8 tier distribution).
- **Algorithm:** `RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)`.

---

## 🎯 Model Evaluation & Performance

A comprehensive evaluation protocol was executed, assessing both single-split holdout test performance and 5-fold cross-validation stability against a baseline classifier.

### 1. Performance Overview

| Evaluation Metric | Baseline (Dummy Classifier) | Random Forest Model | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **Test Accuracy** | **33.33%** (0.3333) | **79.17%** (0.7917) | **+45.84%** |
| **Test Macro F1-Score** | **16.67%** (0.1667) | **79.36%** (0.7936) | **+62.69%** |
| **Test Balanced Accuracy** | **33.33%** (0.3333) | **79.17%** (0.7917) | **+45.84%** |
| **5-Fold CV Mean Macro F1** | — | **88.33%** (0.8833) | — |
| **5-Fold CV Std Deviation** | — | **±6.18%** (0.0618) | — |

> **Key Takeaway:** The model demonstrates strong predictive power, achieving **79.17% test accuracy** on the unseen holdout set and an average of **88.33% Macro F1** across 5-fold cross-validation, substantially beating naive baseline assignment.

---

### 2. Detailed Classification Report (Test Set, N = 24)

```text
              precision    recall  f1-score   support

        High       0.88      0.88      0.88         8
         Low       0.86      0.75      0.80         8
      Medium       0.67      0.75      0.71         8

    accuracy                           0.79        24
   macro avg       0.80      0.79      0.79        24
weighted avg       0.80      0.79      0.79        24
```

- **High Tier:** High precision ($88\%$) and recall ($88\%$), demonstrating clean separation of top-performing instructors.
- **Low Tier:** High precision ($86\%$) with $75\%$ recall, effectively capturing instructors needing support.
- **Medium Tier:** F1-score of $71\%$; represents borderline instructors lying along tercile decision boundaries.

---

### 3. Confusion Matrix Breakdown

The test set confusion matrix demonstrates an essential property: **errors are strictly adjacent**.

```text
                Predicted Low    Predicted Medium    Predicted High
Actual Low            6                 2                  0
Actual Medium         1                 6                  1
Actual High           0                 1                  7
```

- **Zero Severe Misclassifications:**
  - $\text{Actual Low} \to \text{Predicted High} = \mathbf{0}$
  - $\text{Actual High} \to \text{Predicted Low} = \mathbf{0}$
- **Adjacent Boundary Errors Only:**
  - 2 Low instructors were classified as Medium.
  - 1 Medium instructor was classified as Low, and 1 Medium was classified as High.
  - 1 High instructor was classified as Medium.
- This boundary behavior confirms that the model preserves the ordinal hierarchy of effectiveness.

---

### 4. 5-Fold Stratified Cross-Validation Results

To ensure results were not an artifact of a single test split ($N = 24$), a 5-fold stratified cross-validation was evaluated across the full dataset:

$$\text{CV F1 Scores} = [0.9165,\ 0.8381,\ 0.9165,\ 0.7871,\ 0.9582]$$
$$\text{Mean CV Macro F1} = \mathbf{0.8833} \quad (\mathbf{88.33\%})$$
$$\text{Standard Deviation} = \mathbf{0.0618} \quad (\mathbf{\pm 6.18\%})$$

---

### 5. Feature Importance Analysis

Both tree-based Gini impurity importance and test-set permutation feature importance were evaluated:

| Feature Name | Dimension | Gini Importance | Permutation Importance (Mean ± Std) |
| :--- | :--- | :---: | :---: |
| `completion_rate` | Outcomes | **0.2971** (29.71%) | 0.0264 ± 0.0645 |
| `dropout_rate` | Outcomes | **0.2740** (27.40%) | 0.0558 ± 0.0522 |
| `feedback_response_rate` | Feedback | **0.1398** (13.98%) | 0.0557 ± 0.0326 |
| `avg_score_improvement` | Outcomes | **0.0756** (7.56%) | 0.0177 ± 0.0377 |
| `avg_quiz_score` | Outcomes | **0.0669** (6.69%) | 0.0075 ± 0.0435 |
| `avg_feedback_score` | Feedback | **0.0648** (6.48%) | 0.0051 ± 0.0412 |
| `avg_watch_time` | Engagement | **0.0370** (3.70%) | 0.0068 ± 0.0385 |
| `assignment_submission_rate` | Engagement | **0.0229** (2.29%) | 0.0039 ± 0.0256 |
| `forum_activity_rate` | Engagement | **0.0219** (2.19%) | -0.0028 ± 0.0142 |

**Insights:**
1. **Retention Dominance:** `completion_rate` and `dropout_rate` jointly account for $>57\%$ of the model's split decisions.
2. **The Power of Coverage:** `feedback_response_rate` is the 3rd most influential feature ($14\%$). Instructors with high engagement tend to generate higher voluntary response rates.
3. **Engagement Consistency:** Granular engagement features (watch time, forum activity) showed smaller isolated importances, reflecting moderate baseline variance across batches.

---

### 6. Robustness & Collinearity Ablation Analysis

Because `completion_rate` and `dropout_rate` have an extreme correlation ($-0.99$), ablation tests were conducted to evaluate stability when removing collinear inputs:

| Model Variant | Predictor Count | Features Used | Mean CV Macro F1 | Std CV Macro F1 |
| :--- | :---: | :--- | :---: | :---: |
| **Full Baseline Model** | 9 | All features | **0.8833** | 0.0618 |
| **Without Dropout Rate** | 8 | Removed `dropout_rate` | **0.8924** | 0.0721 |
| **Without Outcomes Retention** | 7 | Removed `completion_rate` & `dropout_rate` | **0.8763** | 0.0515 |

- **Finding:** Removing `dropout_rate` slightly *improved* cross-validated F1 to **89.24%**, confirming that the model suffers no informational loss when one redundant retention metric is eliminated.
- Even when **both** completion and dropout rates were excluded, the model maintained an **87.63% Macro F1**, demonstrating that engagement, quiz gains, and feedback metrics contain strong independent predictive power.

---

## 💡 Practical EdTech Applications

1. **Automated Early Warning System:**
   - Detect mid-course drops in engagement or quiz improvement to alert academic coaches before batch completion.
2. **Targeted Faculty Development & Mentorship:**
   - Instead of a single grade, instructors receive dimensional diagnostics. For instance, an instructor with high outcome scores but low engagement can receive training on interactive pedagogy.
3. **Curriculum & Content Diagnostics:**
   - Discrepancies between low watch time and high quiz scores can reveal redundant lecture materials or uncalibrated assessments.

---

## 🚀 Project Structure & Execution

### Directory Layout
```text
Instructor Effectiveness/
│
├── instructor_effectiveness_dataset_2000_rows.xlsx  # 2,000 batch dataset
├── Instructor_Effectiveness_Modeling.ipynb         # Jupyter Notebook with full analysis & code
└── README.md                                       # Comprehensive project documentation
```

### Environment Requirements
- Python 3.10+
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.2.0`
- `openpyxl >= 3.1.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`

### Quick Start
Install the dependencies and run the notebook:
```bash
pip install pandas numpy scikit-learn openpyxl matplotlib seaborn jupyter
jupyter notebook Instructor_Effectiveness_Modeling.ipynb
```

---

## 📜 Authors & Citation
- **Project:** Instructor Effectiveness Modeling Assignment
- **Domain:** EdTech Learning Analytics, Faculty Development & Predictive Modeling
- **Date:** September 2026

