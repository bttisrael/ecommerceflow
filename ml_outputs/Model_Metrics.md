# Model Metrics — Delay Risk Prediction

Best model: **LogisticRegression**  
Best threshold: **0.190**  
Train rows: **80,000**  
Test rows: **20,000**

                name  roc_auc  average_precision  accuracy  precision   recall       f1  threshold  selection_score  tn    fp  fn   tp  business_score_per_order
  LogisticRegression 0.733363           0.550170   0.28545   0.276005 0.997069 0.432334   0.189768         0.733603 267 14275  16 5442                  10.96575
HistGradientBoosting 0.730030           0.544492   0.27310   0.272955 1.000000 0.428852   0.065292         0.732349   4 14538   0 5458                  10.92850
            LightGBM 0.727060           0.541757   0.27320   0.272982 1.000000 0.428886   0.088605         0.731064   6 14536   0 5458                  10.93000
        RandomForest 0.726926           0.540743   0.27480   0.273419 1.000000 0.429426   0.163058         0.730893  38 14504   0 5458                  10.95400
          ExtraTrees 0.726326           0.539871   0.27400   0.273178 0.999817 0.429111   0.156690         0.730459  23 14519   1 5457                  10.93275
             XGBoost 0.726523           0.537433   0.28505   0.275939 0.997435 0.432286   0.069707         0.729482 257 14285  14 5444                  10.97825

## Classification Report
```text
              precision    recall  f1-score   support

           0       0.94      0.02      0.04     14542
           1       0.28      1.00      0.43      5458

    accuracy                           0.29     20000
   macro avg       0.61      0.51      0.23     20000
weighted avg       0.76      0.29      0.14     20000

```