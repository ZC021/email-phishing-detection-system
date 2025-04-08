import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import glob
import logging
from tqdm import tqdm

# 프로젝트 루트 디렉토리 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ml_model import PhishingDetectionModel
from utils.feature_extractor import FeatureExtractor
from utils.email_processor import EmailProcessor
from config import Config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('train_model')

def load_data(data_path):
    """
    CSV 형식의 데이터를 로드합니다.
    
    Args:
        data_path (str): 데이터 파일 경로
        
    Returns:
        pd.DataFrame: 로드된 데이터프레임
    """
    return pd.read_csv(data_path)

def process_emails(emails_dir, is_phishing=False, limit=None):
    """
    디렉토리에서 이메일 파일을 처리하여 특성을 추출합니다.
    
    Args:
        emails_dir (str): 이메일 파일이 있는 디렉토리 경로
        is_phishing (bool): 이메일이 피싱인지 여부
        limit (int, optional): 처리할 최대 이메일 수
        
    Returns:
        pd.DataFrame: 추출된 특성을 포함하는 데이터프레임
    """
    logger.info(f"{'피싱' if is_phishing else '정상'} 이메일 처리 시작: {emails_dir}")
    
    # 이메일 처리기 및 특성 추출기 초기화
    email_processor = EmailProcessor()
    feature_extractor = FeatureExtractor()
    
    # 이메일 파일 목록 수집
    email_files = glob.glob(os.path.join(emails_dir, "*.eml"))
    email_files += glob.glob(os.path.join(emails_dir, "*.txt"))
    
    if limit:
        email_files = email_files[:limit]
    
    all_features = []
    
    # 파일 처리 및 특성 추출
    for file_path in tqdm(email_files, desc=f"{'피싱' if is_phishing else '정상'} 이메일 처리"):
        try:
            # 이메일 내용 및 헤더 추출
            content, headers = email_processor.process_email_file(file_path)
            
            # 특성 추출
            features = feature_extractor.extract_features(content, headers)
            
            # 피싱 여부 레이블 추가
            features['is_phishing'] = 1 if is_phishing else 0
            
            all_features.append(features)
        except Exception as e:
            logger.error(f"파일 처리 중 오류 발생: {file_path}, 오류: {e}")
    
    # 데이터프레임으로 변환
    if all_features:
        df = pd.DataFrame(all_features)
        logger.info(f"처리된 이메일 수: {len(df)}")
        return df
    else:
        logger.warning("처리된 이메일이 없습니다.")
        return pd.DataFrame()

def train_and_evaluate_model(X_train, X_test, y_train, y_test, text_features=None):
    """
    모델을 훈련하고 평가합니다.
    
    Args:
        X_train (pd.DataFrame): 훈련 특성
        X_test (pd.DataFrame): 테스트 특성
        y_train (pd.Series): 훈련 레이블
        y_test (pd.Series): 테스트 레이블
        text_features (list, optional): 텍스트 특성 목록
        
    Returns:
        tuple: (훈련된 모델, 평가 결과 딕셔너리)
    """
    logger.info("모델 훈련 시작")
    
    # 모델 초기화 및 훈련
    model = PhishingDetectionModel()
    model.train(X_train, y_train, text_features=text_features)
    
    # 예측
    y_pred = model.predict(X_test, text_features=text_features)
    y_proba = model.predict_proba(X_test, text_features=text_features)[:, 1]
    
    # 평가 지표 계산
    accuracy = accuracy_score(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # ROC 곡선을 위한 데이터
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # 결과 저장
    results = {
        'accuracy': accuracy,
        'classification_report': clf_report,
        'confusion_matrix': conf_matrix,
        'roc_data': {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auc': roc_auc
        }
    }
    
    logger.info(f"모델 정확도: {accuracy:.4f}, AUC: {roc_auc:.4f}")
    logger.info(f"분류 리포트:\n{classification_report(y_test, y_pred)}")
    
    return model, results

def visualize_results(results, output_dir):
    """
    평가 결과를 시각화하고 저장합니다.
    
    Args:
        results (dict): 평가 결과
        output_dir (str): 출력 디렉토리
    """
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 혼동 행렬 시각화
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        results['confusion_matrix'],
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['정상', '피싱'],
        yticklabels=['정상', '피싱']
    )
    plt.xlabel('예측')
    plt.ylabel('실제')
    plt.title('혼동 행렬')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    
    # ROC 곡선 시각화
    plt.figure(figsize=(8, 6))
    roc_data = results['roc_data']
    plt.plot(
        roc_data['fpr'],
        roc_data['tpr'],
        color='darkorange',
        lw=2,
        label=f'ROC 곡선 (AUC = {roc_data["auc"]:.2f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('거짓 양성 비율')
    plt.ylabel('참 양성 비율')
    plt.title('ROC 곡선')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    
    # 분류 결과 지표 저장
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"정확도: {results['accuracy']:.4f}\n\n")
        clf_report = results['classification_report']
        f.write("분류 리포트:\n")
        f.write(f"              정밀도    재현율  F1-점수   지원\n")
        f.write(f"         0    {clf_report['0']['precision']:.2f}     {clf_report['0']['recall']:.2f}     {clf_report['0']['f1-score']:.2f}     {clf_report['0']['support']}\n")
        f.write(f"         1    {clf_report['1']['precision']:.2f}     {clf_report['1']['recall']:.2f}     {clf_report['1']['f1-score']:.2f}     {clf_report['1']['support']}\n\n")
        f.write(f"    정확도                        {clf_report['accuracy']:.2f}     {clf_report['macro avg']['support']}\n")
        f.write(f"   매크로 평균    {clf_report['macro avg']['precision']:.2f}     {clf_report['macro avg']['recall']:.2f}     {clf_report['macro avg']['f1-score']:.2f}     {clf_report['macro avg']['support']}\n")
        f.write(f"   가중치 평균    {clf_report['weighted avg']['precision']:.2f}     {clf_report['weighted avg']['recall']:.2f}     {clf_report['weighted avg']['f1-score']:.2f}     {clf_report['weighted avg']['support']}\n")

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='이메일 피싱 탐지 모델 훈련')
    
    # 데이터 소스 옵션
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--csv', help='학습 데이터 CSV 파일 경로')
    data_group.add_argument('--email-dirs', nargs=2, help='이메일 디렉토리 경로 (정상 피싱)')
    
    # 추가 옵션
    parser.add_argument('--test-size', type=float, default=0.2, help='테스트 세트 비율 (기본값: 0.2)')
    parser.add_argument('--limit', type=int, help='처리할 최대 이메일 수')
    parser.add_argument('--output-dir', default='output', help='결과 저장 디렉토리 (기본값: output)')
    parser.add_argument('--model-path', help='모델 저장 경로 (기본값: config에서 정의)')
    parser.add_argument('--vectorizer-path', help='벡터라이저 저장 경로 (기본값: config에서 정의)')
    
    args = parser.parse_args()
    
    # 설정 로드
    config = Config()
    model_path = args.model_path or config.MODEL_PATH
    vectorizer_path = args.vectorizer_path or config.VECTORIZER_PATH
    
    # 데이터 로드
    if args.csv:
        logger.info(f"CSV 파일에서 데이터 로드: {args.csv}")
        data = load_data(args.csv)
        X = data.drop(columns=['is_phishing'])
        y = data['is_phishing']
    else:
        logger.info(f"이메일 디렉토리에서 데이터 처리")
        normal_dir, phishing_dir = args.email_dirs
        
        normal_df = process_emails(normal_dir, is_phishing=False, limit=args.limit)
        phishing_df = process_emails(phishing_dir, is_phishing=True, limit=args.limit)
        
        # 데이터 병합
        data = pd.concat([normal_df, phishing_df], ignore_index=True)
        
        # 특성과 레이블 분리
        X = data.drop(columns=['is_phishing'])
        y = data['is_phishing']
    
    # 훈련/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info(f"데이터 분할: 훈련 {len(X_train)}개, 테스트 {len(X_test)}개")
    
    # 모델 훈련 및 평가
    model, results = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    
    # 결과 시각화
    visualize_results(results, args.output_dir)
    
    # 모델 저장
    model.save_model(model_path, vectorizer_path)
    logger.info(f"모델 저장 완료: {model_path}")
    
    logger.info("훈련 완료!")

if __name__ == "__main__":
    main()
