import os
import uuid
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import pandas as pd
import joblib

from config import Config
from utils.email_processor import EmailProcessor
from utils.feature_extractor import FeatureExtractor
from models.ml_model import PhishingDetectionModel

app = Flask(__name__)
app.config.from_object(Config)

# 필요한 디렉터리 생성
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.dirname(app.config['MODEL_PATH']), exist_ok=True)

# 이메일 처리 및 특성 추출 도구 초기화
email_processor = EmailProcessor()
feature_extractor = FeatureExtractor()

# 머신러닝 모델 로드 (모델이 있는 경우)
try:
    ml_model = PhishingDetectionModel(
        model_path=app.config['MODEL_PATH'],
        vectorizer_path=app.config['VECTORIZER_PATH']
    )
    model_loaded = True
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")
    ml_model = None
    model_loaded = False

def allowed_file(filename):
    """파일 확장자가 허용되는지 검사"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """메인 페이지"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/analyze', methods=['POST'])
def analyze_email():
    """이메일 분석 처리"""
    result = {
        'success': False,
        'is_phishing': False,
        'probability': 0,
        'features': {},
        'message': ''
    }
    
    try:
        if 'email_file' in request.files:
            # 파일 업로드 처리
            email_file = request.files['email_file']
            
            if email_file.filename == '':
                result['message'] = '파일이 선택되지 않았습니다.'
                return jsonify(result)
                
            if email_file and allowed_file(email_file.filename):
                # 임시 파일명 생성
                temp_filename = str(uuid.uuid4()) + '.' + email_file.filename.rsplit('.', 1)[1].lower()
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                email_file.save(file_path)
                
                # 이메일 처리
                content, headers = email_processor.process_email_file(file_path)
                
                # 임시 파일 삭제
                os.remove(file_path)
            else:
                result['message'] = '허용되지 않는 파일 형식입니다.'
                return jsonify(result)
        elif 'email_content' in request.form and request.form['email_content'].strip():
            # 이메일 내용 텍스트로 처리
            email_text = request.form['email_content']
            content, headers = email_processor.process_email_text(email_text)
        else:
            result['message'] = '이메일 파일이나 내용이 제공되지 않았습니다.'
            return jsonify(result)

        # 특성 추출
        features = feature_extractor.extract_features(content, headers)
        result['features'] = features
        
        # 모델이 로드되었는지 확인
        if not model_loaded or ml_model is None:
            result['message'] = '모델이 로드되지 않았습니다. 먼저 모델을 훈련해 주세요.'
            result['success'] = True  # 특성은 추출했으므로 부분적 성공
            return jsonify(result)
            
        # 예측 수행
        feature_df = pd.DataFrame([features])
        probability = ml_model.predict_proba(feature_df)[0][1]  # 피싱일 확률
        is_phishing = probability > 0.5
        
        # 결과 설정
        result['success'] = True
        result['is_phishing'] = bool(is_phishing)
        result['probability'] = float(probability)
        result['message'] = '분석이 완료되었습니다.'
        
    except Exception as e:
        result['message'] = f'분석 중 오류가 발생했습니다: {str(e)}'
        
    return jsonify(result)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """모델 훈련 페이지 및 처리"""
    if request.method == 'POST':
        # 여기에 모델 훈련 로직 추가 (데이터 세트 업로드 및 훈련)
        pass
    return render_template('train.html')

@app.route('/about')
def about():
    """프로젝트 소개 페이지"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)