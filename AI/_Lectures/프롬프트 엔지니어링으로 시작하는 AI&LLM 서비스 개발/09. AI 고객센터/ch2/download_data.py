import requests
from bs4 import BeautifulSoup


def get_urls():
    url = "https://fastcampus.co.kr/data_online_llmservice"

    # 웹 페이지 요청 및 응답 받기
    response = requests.get(url)

    # 응답 받은 HTML을 BeautifulSoup을 이용하여 파싱
    soup = BeautifulSoup(response.text, 'html.parser')

    # HTML에서 모든 <img> 태그를 찾음
    img_list = soup.find_all('img')

    # <img> 태그에서 src 속성을 추출하여 리스트로 저장 (src 속성이 없는 경우 제외)
    url_list = [tag.get('src') for tag in img_list if tag.get('src')]

    # 이미지 URL 리스트 반환
    return url_list


if __name__ == '__main__':
    get_urls()