# 야놀자 리뷰 요약

## 문제 조건 확인

문제 조건을 설정하는 이유는 요약 품질이 저하될 수 있고, AI 모델이 감당할 수 없는 입력이 들어올 수 있기 때문에, 전처리가 필요하다.

 - 문제 조건1
    - 모텔은 최근 3개월 데이터 사용
    - 호텔/펜션/게스트하우스는 최근 6개월 후기 데이터 사용
 - 문제 조건2
    - 후기는 3개 이상
    - 후기 글의 합이 90자 이상
 - 문제 조건3
    - 후기가 많을 경우 최근 작성 기준으로 우선 요약

## 데이터 확보 방법론

 - __데이터 크롤링 / API__
    - 태스크에 맞춰서 데이터 확보
    - 상업적 목적의 경우 고려해야 할 사항이 많아짐
 - __각종 데이터 허브__
    - AI 허브(국내)
    - 허깅페이스(글로벌)
    - 데이터 구매
 - __직접 생성__
    - 휴먼 레이블링
    - ChatGPT API를 활용한 합성 데이터 생성

## 실습 환경 준비

```bash
# Homebrew 설치
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치
brew install python3@3.12

# Python Alias 지정
vi ~/.zshrc
alias python='python3'
alias pip='pip3'

# Visual Studio 설치 -> Python Extension, Jupyter 설치
brew install --cask visual-studio-code

# 가상 환경 생성
python3 -m venv .venv
source .venv/bin/activate
pip3 list

python3 -m ipykernel install --user --name .venv --display-name ".venv"
```

## 데이터 크롤링 실습

 - `라이브러리 설치`
```bash
source .venv/bin/activate
pip install bs4 selenium
```

 - `크롤링 코드`
```python
import json
import sys
import time

from bs4 import BeautifulSoup
from selenium import webdriver


def crawl_yanolja_reviews(name, url):
    review_list = []  # 리뷰 데이터를 저장할 리스트

    driver = webdriver.Chrome()  # Selenium을 이용하여 Chrome 웹드라이버 실행
    driver.get(url)  # 지정된 URL로 이동

    time.sleep(3)  # 페이지 로딩을 위해 3초 대기

    scroll_count = 20  # 페이지 스크롤 횟수
    for i in range(scroll_count):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')  # 페이지 최하단으로 스크롤
        time.sleep(2)  # 스크롤 후 2초 대기 (데이터 로딩을 기다리기 위함)

    html = driver.page_source  # 현재 페이지의 HTML 소스 코드 가져오기
    soup = BeautifulSoup(html, 'html.parser')  # BeautifulSoup을 이용해 HTML 파싱

    # 리뷰가 포함된 컨테이너 요소들을 선택
    review_containers = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div')
    
    # 리뷰 작성 날짜 요소 선택
    review_date = soup.select('#__next > section > div > div.css-1js0bc8 > div > div > div > div.css-1toaz2b > div > div.css-1ivchjf')

    # 리뷰 컨테이너 개수만큼 반복하며 데이터 수집
    for i in range(len(review_containers)):
        review_text = review_containers[i].find('p', class_='content-text').text  # 리뷰 텍스트 추출

        # 별점 정보를 가진 SVG 요소 선택 (색이 채워진 별 개수 카운트)
        review_stars = review_containers[i].select('path[fill="currentColor"]')
        star_cnt = sum(1 for star in review_stars if not star.has_attr('fill-rule'))  # 유효한 별점 개수 계산

        date = review_date[i].text  # 리뷰 작성 날짜 추출

        # 리뷰 정보를 딕셔너리 형태로 저장
        review_dict = {
            'review': review_text,
            'stars': star_cnt,
            'date': date
        }

        review_list.append(review_dict)  # 리스트에 추가

    # 수집된 리뷰 데이터를 JSON 파일로 저장
    with open(f'./res/{name}.json', 'w') as f:
        json.dump(review_list, f, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    name, url = sys.argv[1], sys.argv[2]  # 실행 시 입력된 인자값(name, url) 받기
    crawl_yanolja_reviews(name=name, url=url)  # 리뷰 크롤링 함수 실행
```

## 모델 개발 실습

 - `환경 변수 및 라이브러리 설치`
```bash
# API 키 환경 변수에 등록
echo "export OPENAI_API_KEY='api key 내용'" >> ~/.zshrc
source ~/.zshrc
echo $OPENAI_API_KEY

# 라이브러리 설치
pip install openai
```

### OPENAI 호출 예시

```python
import os

from openai import OpenAI


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

client = OpenAI(api_key=OPENAI_API_KEY)

completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': 'hi'}],
    temperature=0.0
)

print(completion.choices[0].message.content)
```

### 모델 개발 준비 및 데이터 전처리

```python
# 리뷰 JSON 파일 로드
import json

with open('./res/reviews.json', 'r') as f:
    review_list = json.load(f)

review_list[:3]


# 좋은 리뷰 및 나쁜 리뷰 갯수 확인
# 좋은 평점 = 별 5개
# 나쁜 평점 = 별 4개 이하
good_cnt, bad_cnt = 0, 0
for r in review_list:
    if r['stars'] == 5:
        good_cnt += 1
    else:
        bad_cnt += 1

good_cnt, bad_cnt


# 좋은 리뷰 및 나쁜 리뷰로 구분 후 저장
reviews_good, reviews_bad = [], []
for r in review_list:
    if r['stars'] == 5:
        reviews_good.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')
    else:
        reviews_bad.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')

reviews_bad[:3]


# 리뷰 목록을 하나의 텍스트로 만들기
reviews_good_text = '\n'.join(reviews_good)
reviews_bad_text = '\n'.join(reviews_bad)

reviews_bad_text[:100]
```

### 평가 기준 설정 및 Baseline 모델 개발

 - `전처리 함수 개발`
```python
import datetime  # 날짜 및 시간을 다루기 위한 모듈
from dateutil import parser  # 문자열 형태의 날짜를 datetime 객체로 변환하는 라이브러리
import json  # JSON 데이터를 다루기 위한 모듈

def preprocess_reviews(path='./res/reviews.json'):
    """
    리뷰 데이터를 전처리하여 최근 6개월 이내의 리뷰만 필터링하고,
    별점 5점과 그 외의 리뷰를 각각 분리하여 텍스트로 반환하는 함수.
    
    :param path: 리뷰 데이터가 저장된 JSON 파일의 경로
    :return: 별점 5점 리뷰 문자열, 그 외 리뷰 문자열
    """
    # JSON 파일 읽기
    with open(path, 'r', encoding='utf-8') as f:
        review_list = json.load(f)  # JSON 파일을 파이썬 리스트로 변환

    # 긍정적인 리뷰(별점 5점)와 부정적인 리뷰(그 외) 리스트
    reviews_good, reviews_bad = [], []

    # 현재 날짜 구하기
    current_date = datetime.datetime.now()

    # 6개월 전 기준 날짜 계산
    date_boundary = current_date - datetime.timedelta(days=6*30)

    # 리뷰 데이터를 순회하면서 필터링 및 분류
    for r in review_list:
        review_date_str = r['date']  # 리뷰의 날짜 문자열 가져오기
        
        # 날짜 변환 시 예외 처리 (형식이 잘못된 경우 현재 날짜로 대체)
        try:
            review_date = parser.parse(review_date_str)  # 문자열을 datetime 객체로 변환
        except (ValueError, TypeError):  
            review_date = current_date  # 변환 실패 시 현재 날짜로 설정

        # 6개월 이전의 리뷰는 제외
        if review_date < date_boundary:
            continue

        # 별점이 5점이면 긍정적인 리뷰 리스트에 추가, 그 외는 부정적인 리뷰 리스트에 추가
        if r['stars'] == 5:
            reviews_good.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')
        else:
            reviews_bad.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')

    # 리스트를 줄바꿈('\n')을 기준으로 하나의 문자열로 변환
    reviews_good_text = '\n'.join(reviews_good)
    reviews_bad_text = '\n'.join(reviews_bad)

    return reviews_good_text, reviews_bad_text  # 최종적으로 두 개의 문자열을 반환

# 함수 실행 후 결과 저장
good, bad = preprocess_reviews()

# 긍정적인 리뷰 문자열의 처음 100자 출력 (테스트용)
good[:100]
```

 - `평가용 함수 개발`
    - 평가 기준 설정
        - MT-Bench 논문 기반 Pairwise Comparision (=LLM 기반 평가)
        - 비교하는 방식 vs. 점수 매기는 방식
        - 점수라는게 애매 할 수 있음 (ex. 어느 정도의 요약 품질이 3점인가?)
        - 경험상 점수보다는 비교가 상대적으로 더 정확한 편
```python
# 평가용 함수 작성
def pairwise_eval(reviews, answer_a, answer_b):
    """
    두 개의 AI 보조자가 작성한 숙박 리뷰 요약을 평가하는 함수.

    :param reviews: 사용자가 남긴 숙박 리뷰 텍스트
    :param answer_a: AI Assistant A의 요약 답변
    :param answer_b: AI Assistant B의 요약 답변
    :return: GPT 모델이 선택한 더 나은 답변자 (A, B, C 중 하나)
    """

    # GPT 모델에게 평가를 요청하는 프롬프트 생성
    eval_prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the Korean summaries provided by two
AI assistants to the set of user reviews on accommodations displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[User Reviews]
{reviews}

[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]"""

    # GPT-4o 모델을 사용하여 평가 수행
    completion = client.chat.completions.create(
        model='gpt-4o-2024-05-13',  # OpenAI의 GPT-4o 모델을 사용하여 평가 요청
        messages=[{'role': 'user', 'content': eval_prompt}],  # 사용자 역할의 메시지 전달
        temperature=0.0  # 평가의 일관성을 유지하기 위해 temperature 값을 0으로 설정
    )

    return completion  # 평가 결과 반환
```

 - `Baseline 모델 개발`
```python
# 숙소 리뷰 요약을 위한 기본 프롬프트 설정
PROMPT_BASELINE = f"""아래 숙소 리뷰에 대해 5문장 내로 요약해줘:"""

# 리뷰 데이터 로드 및 전처리 수행 (별점에 따라 긍정/부정 리뷰로 나뉘지만, 여기서는 긍정 리뷰만 사용)
reviews, _ = preprocess_reviews(path='./res/reviews.json')

def summarize(reviews, prompt, temperature=0.0, model='gpt-3.5-turbo-0125'):
    """
    주어진 숙소 리뷰를 GPT 모델을 사용하여 요약하는 함수.

    :param reviews: 요약할 리뷰 텍스트
    :param prompt: 요약을 요청하는 프롬프트 문자열
    :param temperature: 응답의 창의성을 조절하는 값 (0.0이면 보수적인 응답)
    :param model: 사용할 GPT 모델 (기본값: gpt-3.5-turbo-0125)
    :return: 요약된 리뷰 응답 객체
    """
    
    # 프롬프트에 리뷰 내용을 추가하여 모델에 전달할 입력 데이터 생성
    prompt = prompt + '\n\n' + reviews

    # GPT 모델 호출하여 요약 요청
    completion = client.chat.completions.create(
        model=model,  # 사용할 GPT 모델 지정
        messages=[{'role': 'user', 'content': prompt}],  # 사용자 메시지로 프롬프트 전달
        temperature=temperature  # 응답의 랜덤성을 조절 (0.0이면 일관된 요약을 생성)
    )

    return completion  # 생성된 응답 객체 반환

# GPT 모델을 이용해 숙소 리뷰 요약을 생성하고 출력
print(summarize(reviews, PROMPT_BASELINE).choices[0].message.content)
```

### 대규모 평가 스크립트 작성

```python
from tqdm import tqdm  # 진행 상태를 시각적으로 표시하는 라이브러리

def pairwise_eval_batch(reviews, answers_a, answers_b):
    """
    두 개의 AI 모델이 생성한 요약(answers_a, answers_b)에 대해 batch 평가를 수행하는 함수.

    :param reviews: 원본 리뷰 데이터 (같은 리뷰 데이터가 각 비교에 사용됨)
    :param answers_a: 첫 번째 AI 모델이 생성한 요약 리스트
    :param answers_b: 두 번째 AI 모델이 생성한 요약 리스트
    :return: (A가 이긴 횟수, B가 이긴 횟수, 무승부 횟수)
    """

    # 승리 횟수 카운트 변수 초기화
    a_cnt, b_cnt, draw_cnt = 0, 0, 0

    # tqdm을 이용해 진행 상태를 표시하면서 비교 진행
    for i in tqdm(range(len(answers_a))):
        # 개별 평가 수행 (pairwise_eval 함수 호출)
        completion = pairwise_eval(reviews, answers_a[i], answers_b[i])
        
        # 모델의 평가 결과에서 판정 텍스트 추출
        verdict_text = completion.choices[0].message.content

        # 결과에 따라 승리/패배/무승부 카운트 증가
        if '[[A]]' in verdict_text:
            a_cnt += 1  # A 모델이 승리한 경우
        elif '[[B]]' in verdict_text:
            b_cnt += 1  # B 모델이 승리한 경우
        elif '[[C]]' in verdict_text:
            draw_cnt += 1  # 무승부인 경우
        else:
            print('Evaluation Error')  # 예상치 못한 응답이 발생한 경우 에러 메시지 출력

    return a_cnt, b_cnt, draw_cnt  # 최종 결과 반환

# 배치 평가 실행
wins, losses, ties = pairwise_eval_batch(
    reviews, 
    summaries_baseline,  # 첫 번째 AI 모델의 요약 리스트
    [summary_real_20240526 for _ in range(len(summaries_baseline))]  # 두 번째 AI 모델의 동일한 요약 반복
)

# 평가 결과 출력
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')
```

### 모델 고도화

 - `조건들 명시`
```python
# 요약 프롬프트 설정
prompt = f"""당신은 요약 전문가입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

요약 결과는 다음 조건들을 충족해야 합니다:
1. 모든 문장은 항상 존댓말로 끝나야 합니다.
2. 숙소에 대해 소개하는 톤앤매너로 작성해주세요.
  2-1. 좋은 예시
    a) 전반적으로 좋은 숙소였고 방음도 괜찮았다는 평입니다.
    b) 재방문 예정이라는 평들이 존재합니다.
  2-2. 나쁜 예시
    a) 좋은 숙소였고 방음도 괜찮았습니다.
    b) 재방문 예정입니다.
3. 요약 결과는 최소 2문장, 최대 5문장 사이로 작성해주세요.
    
아래 숙소 리뷰들에 대해 요약해주세요:"""

# 평가 횟수 설정
eval_count = 10

# GPT를 이용해 eval_count(10)개의 요약 생성
summaries = [
    summarize(reviews, prompt, temperature=1.0).choices[0].message.content 
    for _ in range(eval_count)
]

# 생성된 요약과 실제 요약(`summary_real_20240526`)을 비교하여 평가 수행
wins, losses, ties = pairwise_eval_batch(
    reviews, 
    summaries,  # 새로 생성한 요약 리스트
    [summary_real_20240526 for _ in range(len(summaries))]  # 비교할 정답 요약 반복 사용
)

# 평가 결과 출력
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')
```

 - `입력 데이터 품질 증가`
```python
import datetime  # 날짜 및 시간을 다루기 위한 모듈
from dateutil import parser  # 문자열 형태의 날짜를 datetime 객체로 변환하는 라이브러리
import json  # JSON 데이터를 다루기 위한 모듈

def preprocess_reviews(path='./res/reviews.json'):
    """
    리뷰 데이터를 전처리하여 최근 6개월 이내의 리뷰만 필터링하고,
    최소 30자 이상인 리뷰만 유지하며, 별점 5점과 그 외 리뷰를 각각 분리하는 함수.

    :param path: 리뷰 데이터가 저장된 JSON 파일의 경로
    :return: 긍정적인 리뷰 문자열(최대 50개), 부정적인 리뷰 문자열(최대 50개)
    """
    
    # JSON 파일 읽기
    with open(path, 'r', encoding='utf-8') as f:
        review_list = json.load(f)  # JSON 파일을 파이썬 리스트로 변환

    # 긍정적인 리뷰(별점 5점)와 부정적인 리뷰(그 외) 리스트
    reviews_good, reviews_bad = [], []

    # 현재 날짜 구하기
    current_date = datetime.datetime.now()

    # 6개월 전 기준 날짜 계산
    date_boundary = current_date - datetime.timedelta(days=6*30)

    # 필터링된 리뷰 개수를 저장할 변수 (30자 미만으로 제외된 개수)
    filtered_cnt = 0  

    # 리뷰 데이터를 순회하면서 필터링 및 분류
    for r in review_list:
        review_date_str = r['date']  # 리뷰의 날짜 문자열 가져오기

        # 날짜 변환 시 예외 처리 (형식이 잘못된 경우 현재 날짜로 대체)
        try:
            review_date = parser.parse(review_date_str)  # 문자열을 datetime 객체로 변환
        except (ValueError, TypeError):  
            review_date = current_date  # 변환 실패 시 현재 날짜로 설정

        # 6개월 이전의 리뷰는 제외
        if review_date < date_boundary:
            continue

        # 리뷰 길이가 30자 미만이면 제외하고 필터링된 개수 증가
        if len(r['review']) < 30:
            filtered_cnt += 1
            continue

        # 별점이 5점이면 긍정적인 리뷰 리스트에 추가, 그 외는 부정적인 리뷰 리스트에 추가
        if r['stars'] == 5:
            reviews_good.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')
        else:
            reviews_bad.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')

    # 긍정 및 부정 리뷰에서 최대 50개만 유지
    reviews_good = reviews_good[:min(len(reviews_good), 50)]
    reviews_bad = reviews_bad[:min(len(reviews_bad), 50)]

    # 리스트를 줄바꿈('\n')을 기준으로 하나의 문자열로 변환
    reviews_good_text = '\n'.join(reviews_good)
    reviews_bad_text = '\n'.join(reviews_bad)

    return reviews_good_text, reviews_bad_text  # 최종적으로 두 개의 문자열을 반환

# 리뷰 데이터 전처리 실행
reviews, _ = preprocess_reviews()


eval_count = 10
summaries = [summarize(reviews, prompt, temperature=1.0, model='gpt-3.5-turbo-0125').choices[0].message.content for _ in range(eval_count)]
wins, losses, ties = pairwise_eval_batch(reviews, summaries, [summary_real_20240526 for _ in range(len(summaries))])
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')
```

 - `Few-Shot Prompting`
```python
# 하나의 숙소 리뷰 파일(`ninetree_pangyo.json`)을 전처리하여 리뷰 데이터 추출
reviews_1shot, _ = preprocess_reviews('./res/ninetree_pangyo.json')

# `reviews_1shot` 데이터를 사용하여 GPT-4 Turbo 모델을 이용해 한 번 요약을 수행
summary_1shot = summarize(
    reviews_1shot, 
    prompt, 
    temperature=0.0,  # 일관된 응답을 위해 temperature를 0으로 설정
    model='gpt-4-turbo-2024-04-09'  # 최신 GPT-4 Turbo 모델 사용
).choices[0].message.content  # 요약된 결과를 추출

# One-shot Learning을 위한 새로운 프롬프트 생성
prompt_1shot = f"""당신은 요약 전문가입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다.

요약 결과는 다음 조건들을 충족해야 합니다:
1. 모든 문장은 항상 존댓말로 끝나야 합니다.
2. 숙소에 대해 소개하는 톤앤매너로 작성해주세요.
  2-1. 좋은 예시
    a) 전반적으로 좋은 숙소였고 방음도 괜찮았다는 평입니다.
    b) 재방문 예정이라는 평들이 존재합니다.
  2-2. 나쁜 예시
    a) 좋은 숙소였고 방음도 괜찮았습니다.
    b) 재방문 예정입니다.
3. 요약 결과는 최소 2문장, 최대 5문장 사이로 작성해주세요.

다음은 리뷰들과 요약 예시입니다.
예시 리뷰들:
{reviews_1shot}  # `ninetree_pangyo.json`에서 가져온 리뷰 데이터
예시 요약 결과:
{summary_1shot}  # 위에서 생성한 한 번 요약된 결과

아래 숙소 리뷰들에 대해 요약해주세요:"""

# 다양한 요약을 생성하기 위해 여러 번 실행하여 `summaries` 리스트에 저장
summaries = [
    summarize(reviews, prompt, temperature=1.0, model='gpt-3.5-turbo-0125').choices[0].message.content 
    for _ in range(eval_count)  # `eval_count`번 반복하여 다양한 요약 생성
]

# 생성된 요약과 실제 정답(`summary_real_20240526`)을 비교하여 평가 수행
wins, losses, ties = pairwise_eval_batch(
    reviews, 
    summaries,  # 새로 생성한 요약 리스트
    [summary_real_20240526 for _ in range(len(summaries))]  # 동일한 정답 요약을 반복하여 비교
)

# 평가 결과 출력
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')


prompt_1shot = f"""당신은 요약 전문가입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다. 다음은 리뷰들과 요약 예시입니다.
예시 리뷰들:
{reviews_1shot}
예시 요약 결과:
{summary_1shot}
    
아래 숙소 리뷰들에 대해 요약해주세요:"""

summaries = [summarize(reviews, prompt_1shot, temperature=1.0, model='gpt-3.5-turbo-0125').choices[0].message.content for _ in range(eval_count)]
wins, losses, ties = pairwise_eval_batch(reviews, summaries, [summary_real_20240526 for _ in range(len(summaries))])
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')



# 첫 번째 샷(One-shot) 학습용 리뷰 데이터 및 요약 생성
reviews_1shot, _ = preprocess_reviews('./res/ninetree_pangyo.json')  # 판교 숙소 리뷰 전처리
summary_1shot = summarize(
    reviews_1shot, 
    prompt, 
    temperature=0.0,  # 일관된 결과를 위해 temperature=0 설정
    model='gpt-4-turbo-2024-04-09'  # 최신 GPT-4 Turbo 모델 사용
).choices[0].message.content  # 요약 결과 추출

# 두 번째 샷(Two-shot) 학습용 리뷰 데이터 및 요약 생성
reviews_2shot, _ = preprocess_reviews('./res/ninetree_yongsan.json')  # 용산 숙소 리뷰 전처리
summary_2shot = summarize(
    reviews_2shot, 
    prompt_1shot,  # 첫 번째 샷의 요약 예시를 포함한 프롬프트 사용
    temperature=0.0,  
    model='gpt-4-turbo-2024-04-09'
).choices[0].message.content  # 요약 결과 추출

# Two-shot Learning을 위한 새로운 프롬프트 생성
prompt_2shot = f"""당신은 요약 전문가입니다. 사용자 숙소 리뷰들이 주어졌을 때 요약하는 것이 당신의 목표입니다. 다음은 리뷰들과 요약 예시입니다.

예시 리뷰들 1:
{reviews_1shot}  # 판교 숙소 리뷰
예시 요약 결과 1:
{summary_1shot}  # 판교 숙소 요약

예시 리뷰들 2:
{reviews_2shot}  # 용산 숙소 리뷰
예시 요약 결과 2:
{summary_2shot}  # 용산 숙소 요약

아래 숙소 리뷰들에 대해 요약해주세요:"""

# 다양한 요약을 생성하기 위해 여러 번 실행하여 `summaries` 리스트에 저장
summaries = [
    summarize(reviews, prompt_2shot, temperature=1.0, model='gpt-3.5-turbo-0125').choices[0].message.content 
    for _ in range(eval_count)  # `eval_count`번 반복하여 다양한 요약 생성
]

# 생성된 요약과 실제 정답(`summary_real_20240526`)을 비교하여 평가 수행
wins, losses, ties = pairwise_eval_batch(
    reviews, 
    summaries,  # 새로 생성한 요약 리스트
    [summary_real_20240526 for _ in range(len(summaries))]  # 동일한 정답 요약을 반복하여 비교
)

# 평가 결과 출력
print(f'Wins: {wins}, Losses: {losses}, Ties: {ties}')
```

## 데모 제작 실습

Gradio는 머신 러닝 및 데이터 과학 애플리케이션을 쉽게 웹 인터페이스로 배포할 수 있도록 도와주는 프레임워크입니다. Python을 사용하여 간단한 코드만으로 인터랙티브한 웹 UI를 생성할 수 있으며, 특히 모델을 배포하고 공유하는 데 유용합니다.

 - __손쉬운 웹 UI 생성__
    - Python 코드 몇 줄로 입력(input)과 출력(output) 요소를 가진 웹 인터페이스를 만들 수 있습니다.
    - HTML/CSS/JS 등의 추가 작업 없이도 GUI를 생성할 수 있음.
 - __다양한 입력/출력 지원__
    - 텍스트, 이미지, 오디오, 비디오, JSON, 파일 업로드 등 다양한 입출력 형식을 지원합니다.
    - 예를 들어, 사용자가 이미지를 업로드하면, 이를 처리하여 결과를 반환하는 UI를 쉽게 만들 수 있습니다.
 - __빠른 배포 및 공유__
    - 로컬에서 실행할 수도 있고, 한 줄의 코드로 공유 가능한 링크(Gradio’s hosted link)를 생성하여 웹 애플리케이션을 공개할 수도 있음.
    - Hugging Face Spaces에 쉽게 배포 가능.

```python
import datetime  # 날짜 및 시간 관련 모듈
import json  # JSON 데이터 처리 모듈
import os  # 환경 변수 접근 모듈
import pickle  # 객체 직렬화 및 역직렬화 모듈
from dateutil import parser  # 문자열 형태의 날짜를 datetime 객체로 변환하는 라이브러리

import gradio as gr  # 웹 기반 UI 생성을 위한 Gradio 라이브러리
from openai import OpenAI  # OpenAI API 호출을 위한 라이브러리

# OpenAI API 키를 환경 변수에서 가져옴
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

# 숙소 이름과 리뷰 데이터 파일 경로 매핑
MAPPING = {
    '인사동': './res/reviews.json',
    '판교': './res/ninetree_pangyo.json',
    '용산': './res/ninetree_yongsan.json'
}

# 사전 학습된 1-shot 프롬프트 로드
with open('./res/prompt_1shot.pickle', 'rb') as f:
    PROMPT = pickle.load(f)


def preprocess_reviews(path='./res/reviews.json'):
    """
    리뷰 데이터를 전처리하여 최근 6개월 이내의 리뷰만 필터링하고,
    최소 30자 이상인 리뷰만 유지하며, 별점 5점과 그 외 리뷰를 각각 분리하는 함수.

    :param path: 리뷰 데이터가 저장된 JSON 파일의 경로
    :return: 긍정적인 리뷰 문자열(최대 50개), 부정적인 리뷰 문자열(최대 50개)
    """
    
    # JSON 파일 읽기
    with open(path, 'r', encoding='utf-8') as f:
        review_list = json.load(f)  # JSON 파일을 파이썬 리스트로 변환

    # 긍정적인 리뷰(별점 5점)와 부정적인 리뷰(그 외) 리스트
    reviews_good, reviews_bad = [], []

    # 현재 날짜 구하기
    current_date = datetime.datetime.now()

    # 6개월 전 기준 날짜 계산
    date_boundary = current_date - datetime.timedelta(days=6*30)

    # 필터링된 리뷰 개수를 저장할 변수 (30자 미만으로 제외된 개수)
    filtered_cnt = 0  

    # 리뷰 데이터를 순회하면서 필터링 및 분류
    for r in review_list:
        review_date_str = r['date']  # 리뷰의 날짜 문자열 가져오기

        # 날짜 변환 시 예외 처리 (형식이 잘못된 경우 현재 날짜로 대체)
        try:
            review_date = parser.parse(review_date_str)  # 문자열을 datetime 객체로 변환
        except (ValueError, TypeError):  
            review_date = current_date  # 변환 실패 시 현재 날짜로 설정

        # 6개월 이전의 리뷰는 제외
        if review_date < date_boundary:
            continue

        # 리뷰 길이가 30자 미만이면 제외하고 필터링된 개수 증가
        if len(r['review']) < 30:
            filtered_cnt += 1
            continue

        # 별점이 5점이면 긍정적인 리뷰 리스트에 추가, 그 외는 부정적인 리뷰 리스트에 추가
        if r['stars'] == 5:
            reviews_good.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')
        else:
            reviews_bad.append('[REVIEW_START]' + r['review'] + '[REVIEW_END]')

    # 긍정 및 부정 리뷰에서 최대 50개만 유지
    reviews_good = reviews_good[:min(len(reviews_good), 50)]
    reviews_bad = reviews_bad[:min(len(reviews_bad), 50)]

    # 리스트를 줄바꿈('\n')을 기준으로 하나의 문자열로 변환
    reviews_good_text = '\n'.join(reviews_good)
    reviews_bad_text = '\n'.join(reviews_bad)

    return reviews_good_text, reviews_bad_text  # 최종적으로 두 개의 문자열을 반환


def summarize(reviews):
    """
    주어진 숙소 리뷰를 OpenAI GPT 모델을 사용하여 요약하는 함수.

    :param reviews: 요약할 리뷰 텍스트
    :return: GPT 모델의 응답 객체
    """
    
    # 사전 학습된 프롬프트와 리뷰 데이터를 결합하여 입력 데이터 생성
    prompt = PROMPT + '\n\n' + reviews

    # OpenAI API 클라이언트 초기화
    client = OpenAI(api_key=OPENAI_API_KEY)

    # GPT 모델을 사용하여 요약 요청
    completion = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',  # GPT-3.5 Turbo 모델 사용
        messages=[{'role': 'user', 'content': prompt}],  # 사용자 메시지로 프롬프트 전달
        temperature=0.0  # 응답의 랜덤성을 제거하여 일관된 요약 생성
    )

    return completion  # 생성된 응답 객체 반환


def fn(accom_name):
    """
    Gradio 인터페이스에서 선택한 숙소의 리뷰를 가져와 요약하는 함수.

    :param accom_name: 사용자가 선택한 숙소 이름 ('인사동', '판교', '용산')
    :return: 높은 평점(긍정) 요약, 낮은 평점(부정) 요약
    """
    
    # 선택한 숙소에 해당하는 리뷰 파일 경로 가져오기
    path = MAPPING[accom_name]

    # 리뷰 데이터를 전처리하여 긍정 리뷰, 부정 리뷰 추출
    reviews_good, reviews_bad = preprocess_reviews(path)

    # 긍정적인 리뷰 요약 수행
    summary_good = summarize(reviews_good).choices[0].message.content

    # 부정적인 리뷰 요약 수행
    summary_bad = summarize(reviews_bad).choices[0].message.content

    return summary_good, summary_bad  # 두 개의 요약 결과 반환


def run_demo():
    """
    Gradio UI를 실행하는 함수.
    """
    
    # Gradio 인터페이스 설정
    demo = gr.Interface(
        fn=fn,  # 사용자가 선택한 숙소에 대한 요약을 수행하는 함수
        inputs=[gr.Radio(['인사동', '판교', '용산'], label='숙소')],  # 사용자 입력 (숙소 선택)
        outputs=[gr.Textbox(label='높은 평점 요약'), gr.Textbox(label='낮은 평점 요약')]  # 요약 결과 출력
    )

    # Gradio 인터페이스 실행 (공유 링크 활성화)
    demo.launch(share=True)


# 스크립트가 직접 실행될 경우 Gradio 데모 실행
if __name__ == '__main__':
    run_demo()

```
