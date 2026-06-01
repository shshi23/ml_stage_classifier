import os
import joblib
import shap
import numpy as np

class ModelManager:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.stage_offset = 1
        self.explainer = None
        self.is_loaded = False

    def load(self):
        try:
            base_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            self.model = joblib.load(os.path.join(base_dir, 'xgb_stage_model.pkl'))
            self.feature_names = joblib.load(os.path.join(base_dir, 'feature_names.pkl'))
            self.explainer = shap.TreeExplainer(self.model)
            self.is_loaded = True
        except Exception as e:
            print(f"Ошибка загрузки модели: {e}")
            self.is_loaded = False

manager = ModelManager()