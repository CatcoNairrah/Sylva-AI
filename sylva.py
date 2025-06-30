# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings('ignore')
from sklearn.feature_selection import SelectFromModel
# Data handling
import pandas as pd
import numpy as np
import joblib
# Preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Metrics & visualization
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Classical ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def create_heart_disease_dataset():
    np.random.seed(42)
    n_samples = 10000

    age = np.random.normal(54, 10, n_samples).astype(int)
    age = np.clip(age, 29, 77)
    sex = np.random.choice([0, 1], n_samples, p=[0.32, 0.68])
    cp = np.random.choice([0, 1, 2, 3], n_samples, p=[0.47, 0.16, 0.29, 0.08])
    trestbps = np.random.normal(131, 17, n_samples).astype(int)
    trestbps = np.clip(trestbps, 94, 200)
    chol = np.random.normal(246, 51, n_samples).astype(int)
    chol = np.clip(chol, 126, 564)
    fbs = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    restecg = np.random.choice([0, 1, 2], n_samples, p=[0.475, 0.520, 0.005])
    thalach = np.random.normal(149, 22, n_samples).astype(int)
    thalach = np.clip(thalach, 71, 202)
    exang = np.random.choice([0, 1], n_samples, p=[0.68, 0.32])
    oldpeak = np.random.exponential(0.8, n_samples)
    oldpeak = np.clip(oldpeak, 0, 6.2)
    slope = np.random.choice([0, 1, 2], n_samples, p=[0.21, 0.14, 0.65])
    ca = np.random.choice([0, 1, 2, 3], n_samples, p=[0.54, 0.29, 0.10, 0.07])
    thal = np.random.choice([0, 1, 2, 3], n_samples, p=[0.02, 0.18, 0.36, 0.44])

    hdl = np.random.normal(55, 15, n_samples) - sex * 5
    hdl = np.clip(hdl, 25, 100)
    ldl = chol * 0.6 + np.random.normal(0, 20, n_samples)
    ldl = np.clip(ldl, 60, 220)
    bmi = np.random.normal(27, 4, n_samples) + sex * 1.5
    bmi = np.clip(bmi, 17, 44)

    diabetes = np.random.choice([0, 1], n_samples, p=[0.88, 0.12])
    stroke = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    smoking = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    inactive = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
    alcohol = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
    family_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    hypertension = ((trestbps > 140) | (age > 60)).astype(int)
    ckd = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])

    risk_score = (
        0.3 * (age > 50) +
        0.2 * sex +
        0.25 * (cp == 0) +
        0.15 * (trestbps > 140) +
        0.1 * (chol > 240) +
        0.05 * fbs +
        0.1 * restecg +
        0.2 * (thalach < 150) +
        0.3 * exang +
        0.25 * (oldpeak > 1) +
        0.1 * slope +
        0.2 * ca +
        0.15 * (thal == 3) +
        0.1 * (hdl < 40) +
        0.15 * (ldl > 130) +
        0.1 * (bmi > 30) +
        0.25 * diabetes +
        0.2 * stroke +
        0.1 * smoking +
        0.15 * inactive +
        0.1 * (alcohol == 2) +
        0.2 * family_history +
        0.15 * hypertension +
        0.2 * ckd
    )

    risk_score += np.random.normal(0, 0.2, n_samples)
    target = (risk_score > 1.8).astype(int)

    data = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal,
        'hdl': hdl.round(1),
        'ldl': ldl.round(1),
        'bmi': bmi.round(1),
        'diabetes': diabetes,
        'stroke': stroke,
        'smoking': smoking, 'inactive': inactive,
        'alcohol': alcohol,
        'family_history': family_history,
        'hypertension': hypertension,
        'ckd': ckd,
        'target': target
    })

    return data

df = create_heart_disease_dataset()
df.to_csv("synthetic_heart_disease.csv", index=False)

df.columns

df.isnull().sum()

df.info()

for i in df.columns:
  print("Unique values",i,df[i].unique())

df.shape

df['target'].value_counts()

X = df.drop('target', axis=1)
y = df.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

from imblearn.over_sampling import SMOTE
smt = SMOTE()
X, y = smt.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train.shape[1]

"""# Grid Search"""

models = {
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {
            "criterion": ["gini", "entropy"],
            "max_depth": [5, 10, 15, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [200, 300],
            "max_depth": [5, 10, 15],
            "class_weight": [None, "balanced"]
        }
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {
            "n_neighbors": [3, 5, 7,11,13,15],
            "weights": ['uniform', 'distance'],
            "p": [1, 2]
        }
    },
    "Extra Trees": {
        "model": ExtraTreesClassifier(random_state=42),
        "params": {
            "n_estimators": [200, 300],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "params": {
            "n_estimators": [200, 300],
            "max_depth": [3, 5],
            "learning_rate": [0.01, 0.1]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(random_state=42),
        "params": {
            "n_estimators": [200, 300],
            "learning_rate": [0.01, 0.1],
            "num_leaves": [15, 31]
        }
    }
}


for name, mp in models.items():
    print(f"\nRunning GridSearchCV for {name}...")
    grid = GridSearchCV(mp["model"], mp["params"], cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best params for {name}: {grid.best_params_}")

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy for best {name}: {acc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=grid.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

    print(classification_report(y_test, y_pred))

lgb_model = LGBMClassifier(
    learning_rate=0.1,
    n_estimators=300,
    num_leaves=31,
    random_state=42
)
lgb_model.fit(X_train, y_train)

xgb_model = XGBClassifier(
    learning_rate=0.1,
    max_depth=3,
    n_estimators=300,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(X_train, y_train)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    class_weight=None,
    random_state=42
)
rf_model.fit(X_train, y_train)

rf_model.fit(X_train, y_train)

rf_probs = rf_model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, rf_probs)
print(f"Final Accuracy: {acc:.4f}")

joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(lgb_model, "lgbm_model.pkl")

joblib.dump(scaler, "scaler.pkl")
