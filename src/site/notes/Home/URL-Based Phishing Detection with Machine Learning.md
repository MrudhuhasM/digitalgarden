---
{"dg-publish":true,"dg-permalink":"url-based-phishing-detection","permalink":"/url-based-phishing-detection/","tags":["machine-learning","cybersecurity","phishing-detection","xgboost","neural-networks"]}
---


# URL-Based Phishing Detection with Machine Learning

Phishing attacks exploit user trust by creating deceptive websites that mimic legitimate ones. Traditional blacklist-based detection fails against novel phishing sites. Machine learning offers an alternative: classify URLs as phishing or legitimate based on structural features, without requiring prior knowledge of specific malicious domains.

This analysis compares six approaches—Logistic Regression, Decision Trees, Random Forest, Gradient Boosting, XGBoost, and Neural Networks—on a dataset of 100,077 URLs characterized by 19 features.

**Kaggle Notebook:** [Advanced URL Analysis for Phishing Detection](https://www.kaggle.com/code/mrudhuhas/advanced-url-analysis-for-phishing-detection)

## Dataset and Features

Each URL is represented by 19 numerical features capturing structural characteristics:

**Length and Character Counts:**
- `url_length`: Total character count
- `n_dots`, `n_hypens`, `n_underline`, `n_slash`: Counts of common structural characters
- `n_questionmark`, `n_equal`, `n_and`: Query parameter indicators
- `n_at`, `n_exclamation`, `n_space`, `n_tilde`, `n_comma`, `n_plus`: Special character frequencies
- `n_asterisk`, `n_hastag`, `n_dollar`, `n_percent`: Rare symbols

**Behavioral Indicators:**
- `n_redirection`: Number of URL redirects

Target variable `phishing` is binary: 0 for legitimate, 1 for phishing.

The dataset contains 36.3% phishing URLs (36,346 phishing vs 63,731 legitimate), providing reasonable class balance.

## Feature Analysis

### URL Length and Dot Frequency

Phishing URLs exhibit longer lengths and higher dot counts compared to legitimate URLs. The distributions require log transformation due to right skew:

```python
# Log-transformed distributions show clear separation
plt.boxplot(np.log(data.loc[data['phishing'] == 0,'url_length']))  # Normal
plt.boxplot(np.log(data.loc[data['phishing'] == 1,'url_length']))  # Phishing
```

Phishing URLs stretch longer, potentially to disguise malicious intent with misleading path components. Extra dots may mimic legitimate subdomains while subtly altering domain structure.

### Special Character Patterns

Character frequency analysis reveals distinct patterns:

**Hyphens, Underscores, Slashes:** Phishing URLs average higher counts. Attackers may use complex structures mimicking legitimate multi-level paths.

**Query Parameters (?, =, &):** Phishing URLs incorporate more question marks, equals signs, and ampersands. Multiple parameters can hide malicious code or create trustworthy appearances.

**Rare Symbols:** Asterisks, hashtags, and dollar signs appear almost exclusively in phishing URLs. While infrequent overall, their presence is a strong signal.

**Percent Signs:** Legitimate URLs use percent encoding more frequently, likely for proper URL encoding of special characters.

### Redirections

The distribution of redirections shows minimal difference between classes. This feature may have limited discriminative power unless combined with other indicators.

## Model Implementations

### Logistic Regression

Baseline linear model with L2 regularization:

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
```

**Performance:**
- Accuracy: 85.7%
- AUC: 0.93
- True Negatives: 17,593 | False Positives: 1,566
- False Negatives: 1,816 | True Positives: 9,049

**Coefficient Analysis:**

Significant positive predictors (increase phishing probability):
- `n_at`: coefficient = 4.36 (p < 0.001)
- `n_asterisk`: coefficient = 2.01 (p < 0.01)
- `n_slash`: coefficient = 1.03 (p < 0.001)
- `n_questionmark`: coefficient = 0.90 (p < 0.001)
- `url_length`: coefficient = 0.07 (p < 0.001)

Significant negative predictors (increase legitimate probability):
- `n_plus`: coefficient = -1.05 (p < 0.001)
- `n_comma`: coefficient = -0.90 (p < 0.001)
- `n_hypens`: coefficient = -0.61 (p < 0.001)
- `n_and`: coefficient = -0.62 (p < 0.001)
- `n_dots`: coefficient = -0.47 (p < 0.001)

Surprisingly, hyphens and dots negatively correlate with phishing despite earlier observations. This suggests multicollinearity or interaction effects—high counts may be common in *both* classes, with other features providing discrimination.

### Decision Tree

Non-linear model capturing feature interactions:

```python
model = DecisionTreeClassifier(
    ccp_alpha=0.0001,
    criterion='gini',
    max_depth=32,
    min_samples_split=5,
    random_state=1
)
```

**Performance:**
- Train Accuracy: 89.0%
- Test Accuracy: 88.7%
- AUC: 0.95
- True Negatives: 17,593 | False Positives: 1,566
- False Negatives: 1,816 | True Positives: 9,049

**Feature Importance:** `url_length`, `n_slash`, and `n_dots` rank highest, confirming their discriminative power.

### Random Forest

Ensemble of decision trees reducing overfitting:

```python
model = RandomForestClassifier(
    n_estimators=80,
    max_depth=18,
    max_features='sqrt',
    min_samples_split=12,
    criterion='gini'
)
```

**Performance:**
- Train Accuracy: 90.8%
- Test Accuracy: 89.5%
- AUC: 0.96
- True Negatives: 17,626 | False Positives: 1,533
- False Negatives: 1,628 | True Positives: 9,237

Improved over single decision tree with fewer false positives (1,533 vs 1,566) and false negatives (1,628 vs 1,816).

### Gradient Boosting

Sequential ensemble building on prior tree errors:

```python
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.01,
    max_depth=12,
    subsample=0.8,
    max_features=0.5,
    random_state=42
)
```

**Performance:**
- Train Accuracy: 91.1%
- Test Accuracy: 89.5%
- AUC: 0.96
- True Negatives: 17,806 | False Positives: 1,353
- False Negatives: 1,803 | True Positives: 9,062

Best false positive rate (1,353), critical for minimizing false alarms on legitimate sites.

### XGBoost

Optimized gradient boosting with regularization:

```python
model = xgb.XGBClassifier(
    objective='binary:logistic',
    colsample_bytree=0.8,
    learning_rate=0.008,
    max_depth=10,
    alpha=10,
    n_estimators=400
)
```

**Performance:**
- Train Accuracy: 89.3%
- Test Accuracy: 89.1%
- AUC: 0.96
- True Negatives: 17,677 | False Positives: 1,482
- False Negatives: 1,787 | True Positives: 9,078

Balanced performance with controlled overfitting through regularization.

### Neural Network

Deep learning with standardized features:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential([
    Input(shape=(19,)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=25, validation_split=0.2)
```

**Performance:**
- Test Accuracy: 89.3%
- True Negatives: 17,706 | False Positives: 1,453
- False Negatives: 1,679 | True Positives: 9,186

**Lowest false negative rate** (1,679), critical for security applications where missed phishing sites have high cost.

## Comparative Analysis

| Model | Test Acc | AUC | FP | FN | Notes |
|-------|----------|-----|----|----|-------|
| Logistic Regression | 85.7% | 0.93 | 1,566 | 1,816 | Baseline, interpretable |
| Decision Tree | 88.7% | 0.95 | 1,566 | 1,816 | Captures non-linearity |
| Random Forest | 89.5% | 0.96 | 1,533 | 1,628 | Reduced overfitting |
| Gradient Boosting | 89.5% | 0.96 | 1,353 | 1,803 | Best FP rate |
| XGBoost | 89.1% | 0.96 | 1,482 | 1,787 | Balanced |
| Neural Network | 89.3% | - | 1,453 | 1,679 | Best FN rate |

**Trade-offs:**
- **Gradient Boosting:** Minimizes false positives, preferred when blocking legitimate sites is costly
- **Neural Network:** Minimizes false negatives, preferred when missing phishing sites is costly
- **Random Forest:** Best overall balance with competitive performance and faster training

## Key Findings

**Effective Features:** URL length, slash count, at-symbol presence, and asterisk count strongly indicate phishing. Query parameter density (?, =, &) also signals potential attacks.

**Model Selection:** Ensemble methods (Random Forest, Gradient Boosting, XGBoost) consistently outperform single models. Neural networks achieve lowest false negative rate at the cost of interpretability.

**Deployment Considerations:** False positive/negative trade-off depends on context. Security-critical applications may prefer neural networks (minimize missed attacks); user-facing systems may prefer gradient boosting (minimize false alarms).

**Limitations:** URL features alone cannot detect sophisticated phishing using legitimate-looking structures. Hybrid approaches combining URL analysis with content inspection and reputation systems would provide stronger defense.

## Conclusion

Machine learning effectively detects phishing URLs through structural feature analysis. Ensemble methods achieve ~90% accuracy with AUC of 0.96. The neural network minimizes false negatives (1,679), while gradient boosting minimizes false positives (1,353).

Feature engineering reveals that attackers create longer URLs with unusual special character distributions. At-symbols, asterisks, and excessive slashes serve as strong phishing indicators.

For production deployment, Random Forest offers the best balance of accuracy, speed, and interpretability. Critical applications should consider neural networks despite reduced interpretability, given their superior ability to detect attacks.

**Full implementation:** [Kaggle Notebook](https://www.kaggle.com/code/mrudhuhas/advanced-url-analysis-for-phishing-detection)
