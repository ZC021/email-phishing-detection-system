# 이메일 피싱 탐지 시스템

이 프로젝트는 머신러닝 기법을 활용하여 이메일 피싱 탐지를 수행하는 Flask 웹 애플리케이션입니다.

## 주요 기능

- 이메일 내용 분석 및 피싱 여부 탐지
- 머신러닝 모델을 활용한 실시간 분석
- 직관적인 웹 인터페이스
- 상세한 분석 결과 제공

## 기술 스택

- Python 3.8+
- Flask
- scikit-learn
- pandas, numpy
- NLTK
- Bootstrap 5

## 설치 방법

```bash
# 저장소 복제
git clone https://github.com/ZC021/email-phishing-detection-system.git
cd email-phishing-detection-system

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 패키지 설치
pip install -r requirements.txt

# 애플리케이션 실행
python app.py
```

## 사용 방법

1. 웹 브라우저에서 `http://127.0.0.1:5000/`에 접속합니다.
2. 분석할 이메일 파일을 업로드하거나 이메일 내용을 붙여넣습니다.
3. "분석" 버튼을 클릭하여 결과를 확인합니다.

## 라이센스

MIT License