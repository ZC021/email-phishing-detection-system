import re
import nltk
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import numpy as np

# NLTK 데이터 다운로드 (첫 실행시 필요)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class FeatureExtractor:
    """이메일에서 피싱 탐지를 위한 특성을 추출하는 클래스"""
    
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
    
    def extract_features(self, email_content, email_headers=None):
        """
        이메일 내용과 헤더에서 특성을 추출합니다.
        
        Args:
            email_content (str): 이메일 본문 내용
            email_headers (dict, optional): 이메일 헤더 정보
            
        Returns:
            dict: 추출된 특성들
        """
        if email_headers is None:
            email_headers = {}
            
        features = {}
        
        # 텍스트 기반 특성
        features.update(self._extract_text_features(email_content))
        
        # URL 기반 특성
        features.update(self._extract_url_features(email_content))
        
        # HTML 기반 특성
        features.update(self._extract_html_features(email_content))
        
        # 헤더 기반 특성
        if email_headers:
            features.update(self._extract_header_features(email_headers))
        
        return features
    
    def _extract_text_features(self, content):
        """텍스트 관련 특성을 추출합니다."""
        features = {}
        
        # 텍스트만 추출 (HTML 제거)
        if bool(BeautifulSoup(content, "html.parser").find()):
            text_content = BeautifulSoup(content, "html.parser").get_text()
        else:
            text_content = content
            
        # 텍스트 길이
        features['text_length'] = len(text_content)
        
        # 단어 수
        words = nltk.word_tokenize(text_content.lower())
        features['word_count'] = len(words)
        
        # 문장 수
        sentences = nltk.sent_tokenize(text_content)
        features['sentence_count'] = len(sentences)
        
        # 평균 문장 길이
        if features['sentence_count'] > 0:
            features['avg_sentence_length'] = features['word_count'] / features['sentence_count']
        else:
            features['avg_sentence_length'] = 0
            
        # 특수문자 비율
        special_char_count = len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', text_content))
        if features['text_length'] > 0:
            features['special_char_ratio'] = special_char_count / features['text_length']
        else:
            features['special_char_ratio'] = 0
            
        # 대문자 비율
        uppercase_count = sum(1 for c in text_content if c.isupper())
        if features['text_length'] > 0:
            features['uppercase_ratio'] = uppercase_count / features['text_length']
        else:
            features['uppercase_ratio'] = 0
            
        # 특정 키워드 존재 여부
        phishing_keywords = ['urgent', 'verify', 'account', 'password', 'bank', 'click', 
                           'confirm', 'update', 'login', 'security', 'alert', 'suspend']
        
        for keyword in phishing_keywords:
            features[f'has_{keyword}'] = 1 if keyword in text_content.lower() else 0
        
        return features
    
    def _extract_url_features(self, content):
        """URL 관련 특성을 추출합니다."""
        features = {}
        
        # URL 추출
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', content)
        urls += re.findall(r'www\.(?:[-\w.]|(?:%[\da-fA-F]{2}))+', content)
        
        features['url_count'] = len(urls)
        
        if urls:
            # URL 길이 통계
            url_lengths = [len(url) for url in urls]
            features['avg_url_length'] = np.mean(url_lengths) if url_lengths else 0
            features['max_url_length'] = max(url_lengths) if url_lengths else 0
            
            # IP 주소 URL 수
            ip_pattern = re.compile(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}')
            features['ip_url_count'] = sum(1 for url in urls if ip_pattern.match(url))
            
            # 도메인 분석
            domains = [urlparse(url).netloc for url in urls if urlparse(url).netloc]
            features['unique_domain_count'] = len(set(domains))
            
            # URL에 '@' 기호 포함 수
            features['at_symbol_in_url_count'] = sum(1 for url in urls if '@' in url)
            
            # URL에 'data:', 'javascript:' 등의 위험한 프로토콜 사용 수
            dangerous_protocols = ['data:', 'javascript:', 'vbscript:']
            features['dangerous_protocol_count'] = sum(
                1 for url in urls if any(proto in url.lower() for proto in dangerous_protocols)
            )
        else:
            features.update({
                'avg_url_length': 0,
                'max_url_length': 0,
                'ip_url_count': 0,
                'unique_domain_count': 0,
                'at_symbol_in_url_count': 0,
                'dangerous_protocol_count': 0
            })
        
        return features
    
    def _extract_html_features(self, content):
        """HTML 관련 특성을 추출합니다."""
        features = {}
        
        # BeautifulSoup으로 HTML 파싱
        soup = BeautifulSoup(content, "html.parser")
        
        # HTML 여부
        features['is_html'] = 1 if bool(soup.find()) else 0
        
        if features['is_html']:
            # 폼 수
            features['form_count'] = len(soup.find_all('form'))
            
            # 입력 필드 수
            features['input_count'] = len(soup.find_all('input'))
            
            # 비밀번호 필드 수
            features['password_input_count'] = len(soup.find_all('input', {'type': 'password'}))
            
            # 숨겨진 필드 수
            features['hidden_input_count'] = len(soup.find_all('input', {'type': 'hidden'}))
            
            # 외부 리소스 수 (이미지, 스크립트, 스타일시트)
            features['external_resource_count'] = (
                len(soup.find_all('img', src=True)) +
                len(soup.find_all('script', src=True)) +
                len(soup.find_all('link', rel='stylesheet'))
            )
            
            # iframe 수
            features['iframe_count'] = len(soup.find_all('iframe'))
            
            # JavaScript 사용 여부
            features['has_javascript'] = 1 if soup.find_all('script') else 0
            
            # 팝업 사용 여부
            js_content = ' '.join([str(script.string) for script in soup.find_all('script') if script.string])
            features['has_popup'] = 1 if re.search(r'window\.open|alert\(', js_content) else 0
            
            # 외부 도메인 링크 비율
            links = soup.find_all('a', href=True)
            if links:
                external_count = 0
                for link in links:
                    href = link['href']
                    if href.startswith('http') and not re.search(r'\.yourdomain\.com', href):
                        external_count += 1
                features['external_link_ratio'] = external_count / len(links) if len(links) > 0 else 0
            else:
                features['external_link_ratio'] = 0
        else:
            # HTML이 아닌 경우 모든 HTML 특성은 0으로 설정
            features.update({
                'form_count': 0,
                'input_count': 0,
                'password_input_count': 0,
                'hidden_input_count': 0,
                'external_resource_count': 0,
                'iframe_count': 0,
                'has_javascript': 0,
                'has_popup': 0,
                'external_link_ratio': 0
            })
        
        return features
    
    def _extract_header_features(self, headers):
        """이메일 헤더에서 특성을 추출합니다."""
        features = {}
        
        # 발신자 도메인과 수신자 도메인이 일치하는지 확인
        if 'From' in headers and 'To' in headers:
            from_domain = headers['From'].split('@')[-1].strip('>')
            to_domain = headers['To'].split('@')[-1].strip('>')
            features['from_to_domain_match'] = 1 if from_domain == to_domain else 0
        else:
            features['from_to_domain_match'] = 0
            
        # SPF, DKIM, DMARC 검증 결과 (헤더에 있는 경우)
        authentication_headers = [
            'Authentication-Results',
            'Received-SPF',
            'DKIM-Signature',
            'DMARC-Filter'
        ]
        
        for auth_header in authentication_headers:
            features[f'has_{auth_header.lower().replace("-", "_")}'] = 1 if auth_header in headers else 0
            
        # 회신 주소가 있는지 확인
        features['has_reply_to'] = 1 if 'Reply-To' in headers else 0
        
        # 회신 주소와 발신자 주소가 다른지 확인
        if features['has_reply_to'] and 'From' in headers:
            reply_to_domain = headers['Reply-To'].split('@')[-1].strip('>')
            from_domain = headers['From'].split('@')[-1].strip('>')
            features['reply_to_from_mismatch'] = 1 if reply_to_domain != from_domain else 0
        else:
            features['reply_to_from_mismatch'] = 0
            
        return features