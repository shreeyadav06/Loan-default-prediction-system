# Executive Summary: Loan Default Prediction System
**Capstone Project — ML Internship (Day 19)**  
*Domain: Retail Banking & Credit Risk Analytics*  
*Target: Cloud Model Execution (Google Colab / Kaggle)*

---

### 1. Project Goal
Credit evaluation in retail banking involves a critical trade-off between customer acquisition and default risk exposure. The objective of this project is to develop an automated, explainable machine learning classification pipeline to predict whether a prospective loan applicant will **repay (`Y`)** or **default (`N`)** on their loan based on personal socio-economic attributes, financial records, and credit bureau compliance.

---

### 2. Key Insights from Exploratory Data Analysis (EDA)
- **Credit History is the Anchor Feature**: Applicants with a clean credit history (`Credit_History == 1.0`) demonstrate an approval rate of **~79.5%**, whereas applicants with past delinquencies or no compliant history (`Credit_History == 0.0`) have an approval rate of only **~8.2%**. This makes credit history the single highest information-gain attribute.
- **Household Repayment Capacity Matters More Than Individual Salary**: Raw `ApplicantIncome` showed high right-skewness and weak standalone correlation with approval. Engineering `TotalIncome = ApplicantIncome + CoapplicantIncome` and computing residual cashflow after monthly installment (`BalanceIncome = TotalIncome - EMI`) established far stronger predictive signals for creditworthiness.
- **Geographic Risk Variations**: Applicants in **Semiurban** areas registered the highest approval rates (~76.8%), compared to Urban (~65.8%) and Rural (~61.5%) areas.
- **Distribution Normalization**: Continuous variables (`LoanAmount`, `TotalIncome`) exhibited severe right skewness with extreme outliers. A logarithmic transformation (`log(1 + x)`) successfully normalized variance and stabilized gradient updates across linear and ensemble algorithms.

---

### 3. Model Benchmarking & Performance Comparison
Four candidate supervised learning models were evaluated on an 80:20 stratified holdout test split:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Baseline)** | 81.3% | 0.800 | 0.976 | 0.880 | 0.772 |
| **Decision Tree Classifier** | 78.0% | 0.812 | 0.894 | 0.851 | 0.718 |
| **Random Forest Classifier (Default)** | 81.3% | 0.816 | 0.952 | 0.878 | 0.785 |
| **Gradient Boosting Classifier** | 79.7% | 0.806 | 0.940 | 0.867 | 0.760 |
| **Tuned Random Forest (GridSearchCV)** | **82.1%** | **0.819** | **0.964** | **0.886** | **0.798** |

---

### 4. Chosen Model & Final Performance Metrics
- **Selected Architecture**: **Tuned Random Forest Classifier** (`n_estimators=100`, `max_depth=5`, `min_samples_split=5`, `criterion='gini'`).
- **Final Accuracy**: **82.1%** on unseen holdout test data.
- **Macro F1-Score**: **0.886** with high sensitivity (Recall: **96.4%**), ensuring creditworthy individuals are rarely misclassified as defaulters while reliably flagging high-risk applications.
- **ROC-AUC**: **0.798**, confirming strong class separation across varying decision threshold cutoffs.

---

### 5. Strategic Recommendations for Lending Institutions
1. **Automated Tier-1 Approval**: Applicants with `Credit_History == 1.0` and `BalanceIncome > 2.5x EMI` should be routed to immediate straight-through digital approval.
2. **Conditional Loan Restructuring**: Applicants with borderline debt-to-income ratios can be granted approval conditionally by extending loan tenure (e.g., from 180 to 360 months) to compress the monthly installment (`EMI`).
3. **High-Risk Protocol**: Any applicant with `Credit_History == 0.0` should be flagged for mandatory collateral pledging or credit co-signer backing prior to manual underwriter review.
