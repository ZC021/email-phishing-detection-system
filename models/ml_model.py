import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

class PhishingDetectionModel:
    """이메일 피싱 탐지를 위한 머신러닝 모델 클래스"""
    
    def __init__(self, model_path=None, vectorizer_path=None):
        """
        모델 초기화
        
        Args:
            model_path (str, optional): 저장된 모델 파일 경로
            vectorizer_path (str, optional): 저장된 벡터라이저 파일 경로
        """
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        
        # 저장된 모델 로드 (있는 경우)
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
        # 저장된 벡터라이저 로드 (있는 경우)
        if vectorizer_path and os.path.exists(vectorizer_path):
            self.vectorizer = joblib.load(vectorizer_path)
    
    def train(self, X, y, text_features=None):
        """
        모델 학습
        
        Args:
            X (pd.DataFrame): 특성 데이터프레임
            y (pd.Series): 레이블 시리즈 (1: 피싱, 0: 정상)
            text_features (list, optional): 텍스트 특성 목록
            
        Returns:
            self: 학습된 모델 인스턴스
        """
        # 텍스트 특성과 수치 특성 분리
        if text_features:
            X_text = X[text_features]
            X_numeric = X.drop(columns=text_features)
            
            # 텍스트 특성에 대한 벡터라이저 생성
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                min_df=5,
                max_df=0.7,
                stop_words='english'
            )
            
            # 텍스트 특성 벡터화
            X_text_vectorized = self.vectorizer.fit_transform(
                X_text.apply(lambda row: ' '.join(row.astype(str)), axis=1)
            )
            
            # 수치 특성 스케일링
            scaler = StandardScaler()
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            
            # 모든 특성 병합
            X_combined = np.hstack([X_numeric_scaled, X_text_vectorized.toarray()])
            
            # 합쳐진 특성 이름 저장
            self.feature_names = list(X_numeric.columns) + self.vectorizer.get_feature_names_out().tolist()
            
            # 랜덤 포레스트 모델 훈련
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_combined, y)
        else:
            # 텍스트 특성이 없는 경우 간단한 모델 훈련
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X, y)
            self.feature_names = list(X.columns)
            
        return self
    
    def predict(self, X, text_features=None):
        """
        새로운 데이터에 대한 예측 수행
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터프레임
            text_features (list, optional): 텍스트 특성 목록
            
        Returns:
            np.array: 예측 레이블 (1: 피싱, 0: 정상)
        """
        if self.model is None:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")
            
        # 텍스트 특성과 수치 특성 분리
        if text_features and self.vectorizer:
            X_text = X[text_features]
            X_numeric = X.drop(columns=text_features)
            
            # 텍스트 특성 벡터화
            X_text_vectorized = self.vectorizer.transform(
                X_text.apply(lambda row: ' '.join(row.astype(str)), axis=1)
            )
            
            # 수치 특성 스케일링
            scaler = StandardScaler()
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            
            # 모든 특성 병합
            X_combined = np.hstack([X_numeric_scaled, X_text_vectorized.toarray()])
            
            return self.model.predict(X_combined)
        else:
            # 텍스트 특성이 없는 경우 직접 예측
            return self.model.predict(X)
    
    def predict_proba(self, X, text_features=None):
        """
        새로운 데이터에 대한 확률 예측 수행
        
        Args:
            X (pd.DataFrame): 예측할 특성 데이터프레임
            text_features (list, optional): 텍스트 특성 목록
            
        Returns:
            np.array: 예측 확률 [정상 확률, 피싱 확률]
        """
        if self.model is None:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")
            
        # 텍스트 특성과 수치 특성 분리
        if text_features and self.vectorizer:
            X_text = X[text_features]
            X_numeric = X.drop(columns=text_features)
            
            # 텍스트 특성 벡터화
            X_text_vectorized = self.vectorizer.transform(
                X_text.apply(lambda row: ' '.join(row.astype(str)), axis=1)
            )
            
            # 수치 특성 스케일링
            scaler = StandardScaler()
            X_numeric_scaled = scaler.fit_transform(X_numeric)
            
            # 모든 특성 병합
            X_combined = np.hstack([X_numeric_scaled, X_text_vectorized.toarray()])
            
            return self.model.predict_proba(X_combined)
        else:
            # 텍스트 특성이 없는 경우 직접 예측
            return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """
        특성 중요도 반환
        
        Returns:
            pd.DataFrame: 특성과 그 중요도를 포함하는 데이터프레임
        """
        if self.model is None:
            raise ValueError("모델이 아직 훈련되지 않았습니다.")
            
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
        else:
            return pd.DataFrame()
    
    def save_model(self, model_path, vectorizer_path=None):
        """
        모델과 벡터라이저를 파일로 저장
        
        Args:
            model_path (str): 모델을 저장할 파일 경로
            vectorizer_path (str, optional): 벡터라이저를 저장할 파일 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
            
        # 모델 디렉토리 생성 (없는 경우)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # 모델 저장
        joblib.dump(self.model, model_path)
        
        # 벡터라이저 저장 (있는 경우)
        if self.vectorizer is not None and vectorizer_path:
            joblib.dump(self.vectorizer, vectorizer_path)