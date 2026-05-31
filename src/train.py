import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'clean_data.csv')
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
BEST_PARAMS = {
    'subsample': 0.8, 'n_estimators': 500, 'min_child_weight': 2,
    'max_depth': 5, 'learning_rate': 0.03, 'gamma': 0.1, 'colsample_bytree': 0.7
}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Стадия', 'id'], errors='ignore')
    y = (df['Стадия']).astype(int)
    
    feature_names = X.columns.tolist()

    base_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(y.unique()),
        eval_metric='mlogloss',
        random_state=42,
        verbosity=0,
        **BEST_PARAMS
    )

    cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    accs, f1s = [], []

    # КРОСС-ВАЛИДАЦИЯ С SMOTE
    print("\nКросс-валидация:")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Применяем SMOTE только к train-фолду
        smote = SMOTE(k_neighbors=3, random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr, y_tr)
        
        fold_model = clone(base_model)
        fold_model.fit(X_tr_res, y_tr_res, verbose=False)
        
        y_pred = fold_model.predict(X_val)
        accs.append(accuracy_score(y_val, y_pred))
        f1s.append(f1_score(y_val, y_pred, average='macro'))
        print(f"  Fold {fold}: Acc={accs[-1]:.3f} | F1={f1s[-1]:.3f}")

    print(f"\nИтоги CV: Accuracy = {np.mean(accs):.3f} | Macro F1 = {np.mean(f1s):.3f}\n")

    # ФИНАЛЬНОЕ ОБУЧЕНИЕ
    smote_final = SMOTE(k_neighbors=3, random_state=42)
    X_res, y_res = smote_final.fit_resample(X, y)
    
    final_model = clone(base_model)
    final_model.fit(X_res, y_res, verbose=False)

    joblib.dump(final_model, os.path.join(MODEL_DIR, 'xgb_stage_model.pkl'))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, 'feature_names.pkl'))

if __name__ == "__main__":
    main()