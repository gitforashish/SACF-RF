# ============================================================
# FULL STANDALONE BASELINE COMPARISON
# REAL DATA EXPERIMENT
# NSL-KDD DATASET
# ============================================================

# ============================================================
# INSTALL LIBRARIES
# ============================================================

!pip install hmmlearn -q

# ============================================================
# IMPORT LIBRARIES
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler
)

from sklearn.decomposition import PCA

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    IsolationForest,
    RandomForestClassifier
)

from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM

from sklearn.metrics import (

    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score

)

from hmmlearn.hmm import GaussianHMM

# ============================================================
# LOAD DATASET
# ============================================================

print("\n================================================")
print("LOADING NSL-KDD DATASET")
print("================================================")

path = "/content/drive/MyDrive/NSLKDD/KDDTestPlus.csv"

df = pd.read_csv(path)

print("\nDataset Shape :", df.shape)

# ============================================================
# LABEL CREATION
# ============================================================

y_true = np.where(
    df['class'] == 'normal',
    0,
    1
)

# ============================================================
# DROP LABEL COLUMN
# ============================================================

df.drop(columns=['class'], inplace=True)

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

for col in df.columns:

    if df[col].dtype == 'object':

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

# ============================================================
# NORMALIZATION
# ============================================================

print("\n================================================")
print("NORMALIZATION STARTED")
print("================================================")

scaler = MinMaxScaler()

X = scaler.fit_transform(df)

print("\n================================================")
print("NORMALIZATION COMPLETED")
print("================================================")

# ============================================================
# PCA REPRESENTATION
# ============================================================

print("\n================================================")
print("PCA REPRESENTATION")
print("================================================")

pca = PCA(n_components=6)

X_pca = pca.fit_transform(X)

print("\nPCA Shape :", X_pca.shape)

# ============================================================
# HMM STATES
# ============================================================

print("\n================================================")
print("HMM OPERATIONAL STATES")
print("================================================")

hmm = GaussianHMM(

    n_components=3,

    covariance_type='diag',

    n_iter=100,

    random_state=42

)

hmm.fit(X_pca)

states = hmm.predict(X_pca)

print("\nStates Generated")

# ============================================================
# ISOLATION FOREST
# ============================================================

print("\n================================================")
print("ISOLATION FOREST")
print("================================================")

iso_base = IsolationForest(

    contamination=0.20,

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

iso_base.fit(X_pca)

iso_labels = iso_base.predict(X_pca)

iso_labels = np.where(
    iso_labels == -1,
    1,
    0
)

anomaly_score = -iso_base.decision_function(X_pca)

print("\nAnomaly Scores Generated")

# ============================================================
# BEHAVIORAL COGNITION ENGINES
# ============================================================

print("\n================================================")
print("BEHAVIORAL COGNITION ENGINES")
print("================================================")

frequency_engine = (

    pd.Series(iso_labels)

    .rolling(10)

    .mean()

    .fillna(0)

)

persistence_engine = (

    pd.Series(anomaly_score)

    .rolling(15)

    .mean()

    .fillna(0)

)

escalation_engine = (

    pd.Series(anomaly_score)

    .diff()

    .abs()

    .fillna(0)

)

precursor_engine = (

    pd.Series(anomaly_score)

    .rolling(5)

    .std()

    .fillna(0)

)

transition_engine = (

    pd.Series(states)

    .diff()

    .abs()

    .fillna(0)

)

state_engine = abs(
    states - np.mean(states)
)

# ============================================================
# ADAPTIVE SEVERITY FUSION
# ============================================================

print("\n================================================")
print("ADAPTIVE SEVERITY FUSION")
print("================================================")

severity = (

    0.35 * frequency_engine +

    0.25 * persistence_engine +

    0.15 * escalation_engine +

    0.10 * precursor_engine +

    0.10 * transition_engine +

    0.05 * state_engine

)

severity = MinMaxScaler().fit_transform(

    severity.values.reshape(-1,1)

).flatten()

severity_score = severity * 100

print("\nSeverity Fusion Completed")

# ============================================================
# SACF FEATURE MATRIX
# ============================================================

print("\n================================================")
print("SACF FEATURE MATRIX")
print("================================================")

sacf_features = pd.DataFrame({

    "severity_score": severity_score,

    "frequency_engine": frequency_engine,

    "persistence_engine": persistence_engine,

    "escalation_engine": escalation_engine,

    "precursor_engine": precursor_engine,

    "transition_engine": transition_engine,

    "state_engine": state_engine,

    "anomaly_score": anomaly_score,

    "hmm_states": states

})

print("\nFeature Shape :", sacf_features.shape)

# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    sacf_features,

    y_true,

    test_size=0.30,

    random_state=42,

    stratify=y_true

)

# ============================================================
# RESULT STORAGE
# ============================================================

results = []

# ============================================================
# 1. ISOLATION FOREST
# ============================================================

print("\n================================================")
print("BASELINE : ISOLATION FOREST")
print("================================================")

iso = IsolationForest(

    contamination=0.20,

    random_state=42,

    n_estimators=200,

    n_jobs=-1

)

iso.fit(X_train)

iso_pred = iso.predict(X_test)

iso_pred = np.where(
    iso_pred == -1,
    1,
    0
)

iso_score = -iso.decision_function(X_test)

results.append({

    'Model': 'Isolation Forest',

    'Precision': precision_score(y_test, iso_pred),

    'Recall': recall_score(y_test, iso_pred),

    'F1-Score': f1_score(y_test, iso_pred),

    'Accuracy': accuracy_score(y_test, iso_pred),

    'ROC-AUC': roc_auc_score(y_test, iso_score)

})

# ============================================================
# 2. LOF
# ============================================================

print("\n================================================")
print("BASELINE : LOF")
print("================================================")

lof = LocalOutlierFactor(

    contamination=0.20,

    novelty=True,

    n_neighbors=20

)

lof.fit(X_train)

lof_pred = lof.predict(X_test)

lof_pred = np.where(
    lof_pred == -1,
    1,
    0
)

lof_score = -lof.decision_function(X_test)

results.append({

    'Model': 'LOF',

    'Precision': precision_score(y_test, lof_pred),

    'Recall': recall_score(y_test, lof_pred),

    'F1-Score': f1_score(y_test, lof_pred),

    'Accuracy': accuracy_score(y_test, lof_pred),

    'ROC-AUC': roc_auc_score(y_test, lof_score)

})

# ============================================================
# 3. ONE-CLASS SVM
# ============================================================

print("\n================================================")
print("BASELINE : ONE-CLASS SVM")
print("================================================")

ocsvm = OneClassSVM(

    kernel='rbf',

    gamma='scale',

    nu=0.20

)

ocsvm.fit(X_train)

ocsvm_pred = ocsvm.predict(X_test)

ocsvm_pred = np.where(
    ocsvm_pred == -1,
    1,
    0
)

ocsvm_score = -ocsvm.decision_function(X_test)

results.append({

    'Model': 'One-Class SVM',

    'Precision': precision_score(y_test, ocsvm_pred),

    'Recall': recall_score(y_test, ocsvm_pred),

    'F1-Score': f1_score(y_test, ocsvm_pred),

    'Accuracy': accuracy_score(y_test, ocsvm_pred),

    'ROC-AUC': roc_auc_score(y_test, ocsvm_score)

})

# ============================================================
# 4. RANDOM FOREST
# ============================================================

print("\n================================================")
print("BASELINE : RANDOM FOREST")
print("================================================")

rf = RandomForestClassifier(

    n_estimators=200,

    random_state=42,

    n_jobs=-1

)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_prob = rf.predict_proba(X_test)[:,1]

results.append({

    'Model': 'Random Forest',

    'Precision': precision_score(y_test, rf_pred),

    'Recall': recall_score(y_test, rf_pred),

    'F1-Score': f1_score(y_test, rf_pred),

    'Accuracy': accuracy_score(y_test, rf_pred),

    'ROC-AUC': roc_auc_score(y_test, rf_prob)

})

# ============================================================
# 5. SACF-RF
# ============================================================

print("\n================================================")
print("PROPOSED : SACF-RF")
print("================================================")

sacf_rf = RandomForestClassifier(

    n_estimators=250,

    max_depth=10,

    random_state=42,

    n_jobs=-1

)

sacf_rf.fit(X_train, y_train)

sacf_pred = sacf_rf.predict(X_test)

sacf_prob = sacf_rf.predict_proba(X_test)[:,1]

results.append({

    'Model': 'SACF-RF (Proposed)',

    'Precision': precision_score(y_test, sacf_pred),

    'Recall': recall_score(y_test, sacf_pred),

    'F1-Score': f1_score(y_test, sacf_pred),

    'Accuracy': accuracy_score(y_test, sacf_pred),

    'ROC-AUC': roc_auc_score(y_test, sacf_prob)

})

# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

comparison_df = pd.DataFrame(results)

print("\n================================================")
print("REAL EXPERIMENT RESULTS")
print("================================================")

print(comparison_df)

# ============================================================
# GRAPH GENERATION
# ============================================================

metrics = [

    'Precision',
    'Recall',
    'F1-Score',
    'Accuracy',
    'ROC-AUC'

]

models = comparison_df['Model']

plt.figure(figsize=(16,8))

x = np.arange(len(metrics))

width = 0.15

for i, model in enumerate(models):

    values = comparison_df.loc[i, metrics].values

    plt.bar(

        x + i * width,

        values,

        width,

        label=model

    )

# ============================================================
# GRAPH STYLE
# ============================================================

plt.xticks(

    x + width * 2,

    metrics,

    fontsize=13

)

plt.ylabel(

    'Score',

    fontsize=14,

    fontweight='bold'

)

plt.xlabel(

    'Evaluation Metrics',

    fontsize=14,

    fontweight='bold'

)

plt.title(

    'Real Experimental Comparison of SACF-RF and Baselines',

    fontsize=18,

    fontweight='bold'

)

plt.legend(

    fontsize=11,

    bbox_to_anchor=(1.01,1),

    loc='upper left'

)

plt.grid(

    axis='y',

    linestyle='--',

    alpha=0.4

)

plt.tight_layout()

# ============================================================
# SAVE FIGURES
# ============================================================

plt.savefig(

    'Figure4_Real_Baseline_Comparison.svg',

    format='svg',

    bbox_inches='tight'

)

plt.savefig(

    'Figure4_Real_Baseline_Comparison.pdf',

    format='pdf',

    bbox_inches='tight'

)

plt.savefig(

    'Figure4_Real_Baseline_Comparison.png',

    dpi=600,

    bbox_inches='tight'

)

# ============================================================
# SHOW GRAPH
# ============================================================

plt.show()

# ============================================================
# COMPLETE
# ============================================================

print("\n================================================")
print("REAL DATA BASELINE COMPARISON COMPLETED")
print("================================================")

print("\nSaved Files:")

print("1. Figure4_Real_Baseline_Comparison.svg")

print("2. Figure4_Real_Baseline_Comparison.pdf")

print("3. Figure4_Real_Baseline_Comparison.png")