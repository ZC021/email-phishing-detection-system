import re
import email
from email.header import decode_header
import base64
import quopri
from bs4 import BeautifulSoup

class EmailProcessor:
    """이메일 파일을 처리하여 내용과 헤더를 추출하는 클래스"""
    
    def __init__(self):
        pass
        
    def process_email_file(self, file_path):
        """
        이메일 파일을 처리하여 내용과 헤더를 추출합니다.
        
        Args:
            file_path (str): 이메일 파일 경로
            
        Returns:
            tuple: (이메일 본문 내용, 헤더 딕셔너리)
        """
        try:
            with open(file_path, 'rb') as f:
                msg = email.message_from_binary_file(f)
                
            # 헤더 추출 및 디코딩
            headers = self._extract_headers(msg)
            
            # 본문 추출
            content = self._extract_content(msg)
            
            return content, headers
            
        except Exception as e:
            print(f"이메일 처리 중 오류 발생: {e}")
            return "", {}
    
    def process_email_text(self, email_text):
        """
        이메일 텍스트 문자열을 처리하여 내용과 헤더를 추출합니다.
        
        Args:
            email_text (str): 이메일 텍스트
            
        Returns:
            tuple: (이메일 본문 내용, 헤더 딕셔너리)
        """
        try:
            msg = email.message_from_string(email_text)
            
            # 헤더 추출 및 디코딩
            headers = self._extract_headers(msg)
            
            # 본문 추출
            content = self._extract_content(msg)
            
            return content, headers
            
        except Exception as e:
            print(f"이메일 처리 중 오류 발생: {e}")
            return "", {}
    
    def _extract_headers(self, msg):
        """이메일 헤더를 추출하고 디코딩합니다."""
        headers = {}
        
        for key in msg.keys():
            value = msg[key]
            # 인코딩된 헤더 디코딩
            decoded_value = ""
            for part, encoding in decode_header(value):
                if isinstance(part, bytes):
                    decoded_part = part.decode(encoding if encoding else 'utf-8', errors='replace')
                else:
                    decoded_part = part
                decoded_value += decoded_part
            
            headers[key] = decoded_value
            
        return headers
    
    def _extract_content(self, msg):
        """이메일 본문 내용을 추출합니다."""
        content = ""
        
        # HTML과 텍스트 본문을 모두 찾기
        html_content = None
        text_content = None
        
        if msg.is_multipart():
            # 멀티파트 이메일 처리
            for part in msg.get_payload():
                content_type = part.get_content_type()
                
                if content_type == 'text/html':
                    html_content = self._decode_part(part)
                elif content_type == 'text/plain':
                    text_content = self._decode_part(part)
                elif part.is_multipart():
                    # 중첩된 멀티파트 처리
                    for subpart in part.get_payload():
                        if subpart.get_content_type() == 'text/html':
                            html_content = self._decode_part(subpart)
                        elif subpart.get_content_type() == 'text/plain':
                            text_content = self._decode_part(subpart)
        else:
            # 단일 파트 이메일 처리
            content_type = msg.get_content_type()
            
            if content_type == 'text/html':
                html_content = self._decode_part(msg)
            elif content_type == 'text/plain':
                text_content = self._decode_part(msg)
        
        # HTML 내용이 있으면 우선 사용, 없으면 텍스트 내용 사용
        if html_content:
            content = html_content
        elif text_content:
            content = text_content
            
        return content
    
    def _decode_part(self, part):
        """이메일 파트의 내용을 디코딩합니다."""
        content = part.get_payload(decode=True)
        charset = part.get_content_charset() or 'utf-8'
        
        try:
            decoded_content = content.decode(charset, errors='replace')
        except:
            # 디코딩 실패 시 기본 utf-8 사용
            decoded_content = content.decode('utf-8', errors='replace')
            
        # Base64 또는 Quoted-Printable로 인코딩된 내용 처리
        transfer_encoding = part.get('Content-Transfer-Encoding', '').lower()
        
        if transfer_encoding == 'base64':
            try:
                decoded_content = base64.b64decode(content).decode(charset, errors='replace')
            except:
                pass
        elif transfer_encoding == 'quoted-printable':
            try:
                decoded_content = quopri.decodestring(content).decode(charset, errors='replace')
            except:
                pass
                
        return decoded_content