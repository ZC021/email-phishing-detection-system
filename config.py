import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class Config:
    """애플리케이션 설정 클래스"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    ALLOWED_EXTENSIONS = {'eml', 'txt', 'html'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB 제한
    MODEL_PATH = os.path.join(os.getcwd(), 'models', 'phishing_model.joblib')
    VECTORIZER_PATH = os.path.join(os.getcwd(), 'models', 'vectorizer.joblib')
    
    # 데이터베이스 설정 (추후 필요시)
    DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///phishing.db'