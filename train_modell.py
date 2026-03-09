"""
Academic Warning Prediction — Real Kaggle Dataset
Target: Academic_Status (0=Normal, 1=Warning, 2=Dropout)
"""
import pandas as pd
import numpy as np
import pickle, re, os, warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (classification_report, accuracy_score,
                              f1_score, confusion_matrix,
                              roc_auc_score)
from sklearn.base import BaseEstimator, TransformerMixin

np.random.seed(42)

# ─────────────────────────────────────────────────────────
# 1.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────
# Vietnamese keyword dictionaries for Advisor_Notes
POSITIVE_KW = ['đúng giờ','chuyên cần','tốt','đáng nể','kịp thời','chăm','tích cực','nỗ lực','tiến bộ']
NEGATIVE_KW = ['nợ môn','không đến lớp','lười biếng','cấm thi','vắng','kém','lo ngại','thôi học','buộc']
DROPOUT_KW  = ['cấm thi','buộc thôi học','thôi học','nợ môn quá','bỏ học']

def keyword_score(text, keywords):
    if pd.isna(text): return 0
    text = text.lower()
    return sum(1 for kw in keywords if kw in text)

def clean_category(s):
    """Normalize messy categorical: lowercase, strip punctuation"""
    if pd.isna(s): return 'unknown'
    return re.sub(r'[.\s]+$', '', str(s).strip().lower())

ENG_ORDER = ['a1','a2','b1','b2','ielts 6.0+']

def engineer_features(df):
    df = df.copy()

    # ── Categorical cleanup
    df['gender']         = (df['Gender'].str.strip().str.lower() == 'nam').astype(int)
    df['club_member']    = (df['Club_Member'].str.strip().str.lower() == 'yes').astype(int)
    df['admission_mode'] = df['Admission_Mode'].apply(clean_category)
    df['english_level']  = df['English_Level'].apply(clean_category)

    # Ordinal encode English level
    eng_map = {e: i for i, e in enumerate(ENG_ORDER)}
    df['english_ord'] = df['english_level'].map(eng_map).fillna(2)

    # Admission mode → label encode after cleanup
    df['admission_enc'] = LabelEncoder().fit_transform(df['admission_mode'])

    # ── Attendance columns: treat -1 as "not enrolled" (NaN for stats)
    att_cols = [c for c in df.columns if c.startswith('Att_')]
    att = df[att_cols].replace(-1, np.nan)

    df['att_mean']      = att.mean(axis=1)
    df['att_min']       = att.min(axis=1)
    df['att_max']       = att.max(axis=1)
    df['att_std']       = att.std(axis=1)
    df['att_count']     = att.notna().sum(axis=1)          # subjects enrolled
    df['att_below_50pct'] = (att < att.shape[1]*0.5).sum(axis=1).fillna(0)  # low-att subjects
    df['att_below_7']   = (att < 7).sum(axis=1).fillna(0)   # fail threshold
    df['att_negative']  = (att < 0).sum(axis=1).fillna(0)   # penalty attendance
    df['att_sum']       = att.sum(axis=1)

    # ── Text features from Advisor_Notes
    df['note_positive'] = df['Advisor_Notes'].apply(lambda x: keyword_score(x, POSITIVE_KW))
    df['note_negative'] = df['Advisor_Notes'].apply(lambda x: keyword_score(x, NEGATIVE_KW))
    df['note_dropout']  = df['Advisor_Notes'].apply(lambda x: keyword_score(x, DROPOUT_KW))
    df['note_len']      = df['Advisor_Notes'].fillna('').apply(len)
    df['note_sentiment'] = df['note_positive'] - df['note_negative']

    # Essay length as proxy for engagement
    df['essay_len']     = df['Personal_Essay'].fillna('').apply(len)

    # ── Numeric originals
    df['tuition_debt']  = df['Tuition_Debt'].fillna(0)
    df['count_f']       = df['Count_F'].fillna(0)
    df['training_score']= df['Training_Score_Mixed']
    df['age']           = df['Age']

    feature_cols = [
        'age','gender','club_member','english_ord','admission_enc',
        'tuition_debt','count_f','training_score',
        'att_mean','att_min','att_max','att_std','att_count',
        'att_below_7','att_negative','att_below_50pct','att_sum',
        'note_positive','note_negative','note_dropout','note_sentiment','note_len',
        'essay_len',
    ]
    return df[feature_cols], feature_cols

# ─────────────────────────────────────────────────────────
# 2.  TRAIN
# ─────────────────────────────────────────────────────────
def train(csv_path='train.csv', out_dir='model'):
    print("=" * 55)
    print("  Academic Warning — Training Pipeline")
    print("=" * 55)

    df = pd.read_csv(csv_path)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    print(f"Target dist:\n{df['Academic_Status'].value_counts().to_string()}\n")

    X, feature_cols = engineer_features(df)
    y = df['Academic_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Pipeline
    model = Pipeline([
        ('imp',   SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('clf',   GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05,
            max_depth=5, subsample=0.8,
            min_samples_leaf=10, random_state=42
        ))
    ])

    # Cross-val
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                 scoring='f1_macro', n_jobs=-1)
    print(f"CV F1-macro: {cv_scores.mean():.4f} ±{cv_scores.std():.4f}")

    model.fit(X_train, y_train)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro')
    try:
        auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='macro')
    except:
        auc = 0.0

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test F1-macro : {f1:.4f}")
    print(f"Test ROC-AUC  : {auc:.4f}")
    print()
    print(classification_report(y_test, y_pred,
          target_names=['Normal','Warning','Dropout']))

    # Feature importance
    fi = pd.Series(model.named_steps['clf'].feature_importances_,
                   index=feature_cols).sort_values(ascending=False)
    print("Top-10 Feature Importances:")
    print(fi.head(10).to_string())

    # ── Save artifacts
    os.makedirs(out_dir, exist_ok=True)
    artifacts = {
        'model': model,
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': acc, 'f1_macro': f1,
            'roc_auc': auc,
            'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
        },
        'feature_importance': fi.to_dict(),
        'class_names': ['Normal (0)', 'Academic Warning (1)', 'Dropout (2)'],
    }
    pkl_path = os.path.join(out_dir, 'academic_warning_model.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"\n✅ Model saved → {pkl_path}")
    return artifacts

if __name__ == '__main__':
    train(csv_path='D:/AII/btvn/train.csv')