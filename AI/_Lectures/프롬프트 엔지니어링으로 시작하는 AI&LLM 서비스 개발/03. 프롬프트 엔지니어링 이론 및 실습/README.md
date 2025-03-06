# 프롬프트 엔지니어링 이론 및 실습

## 1. 프롬프트 엔지니어링 개념 설명

 - Prompt: ChatGPT의 출력을 원하는 방향으로 유도하기 위한 입력 텍스트. Prompt는 보통 질문 또는 지시 형태를 나타냄
 - Role: 역할. 크게는 (1) 사용자를 뜻하는 User (2) ChatGPT를 뜻하는 Assistant 그리고 (3) System이 존재
 - Prompt는 영어로 해야 모델의 제성능을 발휘하는 편
    - ChatGPT, Claude 같은 모델들의 학습 데이터 중 큰 비중이 영어로 추정되기 때문
    - 한글 출력값이 필요하더라도 영어 Prompt를 통해 한글 출력값을 유도하는게 성능이 더 좋을 수 있음
```python
# OpenAI 설치
!pip install openai

# OpenAI 키 등록
from google.colab import userdata
OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

# OpenAI API 호출
from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY
)


completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': '왜 하늘은 하늘색인가요?'}],
    temperature=0.0
)

print(completion.choices[0].message.content)

# System Prompt
# 출력값 지정 (ex. JSON Formatting)
# 페르소나 및 어조 설정
# 외부 정보 주입
# 모델이 지켜야 할 규칙들 설정
completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[
        {'role': 'system', 'content': '당신은 물리학 선생님입니다. 초등학교 5학년한테 설명하듯이 아주 쉽고 친근하게 설명해주세요.'},
        {'role': 'user', 'content': '왜 하늘은 하늘색인가요?'}
    ],
    temperature=0.0
)

print(completion.choices[0].message.content)


completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[
        {'role': 'user', 'content': '당신은 물리학 선생님입니다. 초등학교 5학년한테 설명하듯이 아주 쉽고 친근하게 설명해주세요. 왜 하늘은 하늘색인가요?'}
    ],
    temperature=0.0
)

print(completion.choices[0].message.content)
```

## 2. 프롬프트 개발 주기

 - 명확한 평가 기준 설정
 - 평가를 진행 할 테스트 케이스 선정 (엣지 케이스 포함)
 - Baseline Prompt 선정
    - 고도화 과정 후에 비교를 위한 대조군 설정
 - (반복) 테스트 케이스에 대해 평가 진행
 - (반복) Prompt 수정
 - Prompt 완성
    - Baseline 대비 어떤 지표에서 얼마나 개선되었는지

### 2-1. 명확한 평가 기준 설정

 - `태스크 정의`
    - 요약, Q&A, 코드 생성, 글쓰기 등
    - 각 태스크 별로 사용되는 평가 기준 및 지표들이 다를 수 있음
        - 동일한 요약 안에서도 대화 요약, 문서 요약 등이 존재
        - Q&A도 In-Domain, Out-of-Domain으로 나뉠 수도 있음
    - 어떤 문제를 풀어야하는지 태스크를 구체적으로 명확하게 정의하는 것이 첫 단계
 - `평가 기준 설정`
    - 성능
        - 태스크에서 정확히 어느 정도의 품질이 필요한 지를 정의하는 객관적인 기준
        - 태스크마다 평가 기준 다를 수 있고 여러 개의 평가 지표들이 있을 수 있음
        - 객관식 질문에 대답하는 태스크: 정확도
        - 요약 태스크: 정답과 모델 출력값 간의 문자열 비교 (Exact Match or Partial Match)
    - 응답 속도 (Latency)
        - LLM에서 Latency 정의: Prompt 입력 후 응답 완료까지 걸리는 시간 (주요 용어)
        - 응답 시간. 실시간 및 비실시간 여부에 따라 기준치가 높아지거나 낮아질 수 있음
    - 비용
        - 모델 가격, 사용되는 평균 입력 및 출력 토큰 수, 호출 수 등을 고려한 예상 비용

### 2-2. 평가를 진행할 테스트 케이스 선정

 - `테스트 케이스 확보`
    - 예시 시나리오: 숙소 리뷰 요약
    - 사람이 직접 제작한 N개 정도의 Golden Reference
    - 비용 이슈로 ChatGPT 3.5를 써야한다고 했을 때 더 상위 레벨의 모델 출력 값 (ex. GPT-4)
    - 이미 시중에 존재하는 레이블링 된 데이터 등등
 - `엣지 케이스`
    - 예시 1. 입력이 매우 길거나 매우 짧은 케이스들
    - 예시 2. 토막글이나 실제 숙소에 대한 정보가 별로 없는 리뷰 등의 저품질 리뷰들로만 이루어진 케이스들
    - 예시 3. 입력으로 숙소 리뷰가 N개 이하 들어왔을 때는 요약 결과가 아닌 충분하지 않은 케이스
        - 이런 케이스는 후처리 또는 전처리로 해결하는게 나을 수도 있음

### 2-3. Baseline Prompt 선정

 - 고도화 과정을 확인하기 위한 용도의 Baseline Prompt 선정
 - 정말 단순하고 Naive 한 Prompt
    - 특별히 노력을 거치치 않은 Prompt
    - Prompt Library 중에 하나로 선정

## 3. 평가 기준 설정

 - `MMLU (Massive Multitask Language Understanding) - github, paper`
    - 여러 분야 테스트하는 객관식 시험
    - 참고로 MMLU (5 shot)의 경우 5개의 질문/정답 쌍이 Prompt로 주어졌다는 뜻
    - 깃헙: https://github.com/hendrycks/test
    - 논문: https://arxiv.org/pdf/2009.03300
 - `ARC (Abstraction and Reasoning Corpus) - github, paper`
    - 2차원 pixels grid 주고 특정 문제 해결 ex. 패턴 주고 일부 비워두고 어떤 색깔로 칠 할지 맞추는 문제
    - 깃헙: https://github.com/fchollet/ARC
    - 논문: https://arxiv.org/abs/1911.01547
 - `HellaSwag - website, paper`
    - 문장들 주고 이어지는 마지막 문장들로 가장 적합한 문장들 4개 중 하나 고르는 문제
    - 웹사이트: https://rowanzellers.com/hellaswag/
    - 논문: https://arxiv.org/pdf/1905.07830
 - `TruthfulQA - github, agit`
    - 할루시네이션 측정용 데이터셋이고 주어진 문제에 대한 Accuracy 측정 (문제 유형은 객관식 MC 외에 더 있음)
    - 깃헙: https://github.com/sylinrl/TruthfulQA
    - 논문: https://arxiv.org/abs/2109.07958

### 3-1. 태스크에 적합한 평가 기법 분류

 - `Human Based Method(사람이 평가하는 방법)`
    - 전문가 블라인드 A/B 테스트 --> ELO 리더보드
    - 2가지 답변 중에 더 좋은 답변을 선택하는 방법
    - 명확한 평가 기준
    - 장단점
        - 통제된 환경을 가정 했을 때 사람이 직접 평가한 방법이라 안정적이고 신뢰 할 수 있음
        - 불특정 다수의 경우 약간의 노이즈 발생 가능
        - 전문 도메인의 경우 해당 도메인 전문가가 아닌 일반인이 평가 할 경우 정확도 및 평가 속도가 낮아질 수 있음
 - `Model Based Evaluation(LLM 모델이 평가하는 방법)`
    - GPT-4 같은 Strong LLM을 통해 평가하는 방법 i.e. LLM-as-a-judge
    - Pairwise Comparison: 질문과 답변 2개를 받아 둘 중 어떤 답변이 더 좋은 지 또는 무승부인지 답변
    - Single Answer Grading: 질문과 답변이 있을 때 답변에 점수를 매기는 것
    - Reference-Guided Grading: 예시 답변을 주고 점수를 매기는 것
    - MT-Bench 논문: https://arxiv.org/abs/2306.05685
    - G-Eval 논문: https://arxiv.org/abs/2303.16634
    - 장단점
        - 사람 평가와 어느 정도 유사한 수준의 평가를 내릴 수 있음
        - 평가를 위해 API 호출이 필요한데 평가 데이터가 굉장히 많을 경우 천만원 이상은 금방 넘어 갈 수 있음
 - `Code Based Evaluation(코드로 평가하는 방법)`
    - Accuracy, Precision, Recall...
    - ROUGE: A Package for Automatic Evaluation of Summaries
    - BLEU: a Method for Automatic Evaluation of Machine Translation
    - Exact Match, String Match
    - 장단점
        - 위 방법들과 인력 고용 비용, 모델 호출 비용 등이 없는 무료 평가 방법
        - 태스크에 따라 위 방법들보다 더 정확 할 수도 있고 그러지 않을 수도 있음
        - 정확도 같은 지표를 벗어나 사람한테 적합한 답변을 선택하는데 있어서는 신뢰도가 상대적으로 떨어지는 편

### 3-2. Model Based Evaluation

```python
# API 키 설정
from google.colab import userdata

OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

# 클라이언트 생성
from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY
)
question = '하늘은 왜 하늘색인가요?'


# 답변 1 생성
completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': question}],
    temperature=0.0
)
answer_a = completion.choices[0].message.content
print(answer_a)


# 답변 2 생성
completion = client.chat.completions.create(
    model='gpt-4-1106-preview',
    messages=[{'role': 'user', 'content': question}],
    temperature=0.0
)
answer_b = completion.choices[0].message.content
print(answer_b)
    

# 평가 프롬프트 출처: MT-Bench 논문 https://arxiv.org/pdf/2306.05685.pdf (A. Prompt Templates Figure 5)
prompt = f"""[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two
AI assistants to the user question displayed below. You should choose the assistant that
follows the user’s instructions and answers the user’s question better. Your evaluation
should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
and level of detail of their responses. Begin your evaluation by comparing the two
responses and provide a short explanation. Avoid any position biases and ensure that the
order in which the responses were presented does not influence your decision. Do not allow
the length of the responses to influence your evaluation. Do not favor certain names of
the assistants. Be as objective as possible. After providing your explanation, output your
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
if assistant B is better, and "[[C]]" for a tie.

[User Question]
{question}

[The Start of Assistant A’s Answer]
{answer_a}
[The End of Assistant A’s Answer]

[The Start of Assistant B’s Answer]
{answer_b}
[The End of Assistant B’s Answer]"""

completion = client.chat.completions.create(
    model='gpt-4-turbo-2024-04-09',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)

print(completion.choices[0].message.content)
```

## 4. 잘 알려진 기법들

 - Few-Shot, Chain-of-Thought, 주요 기법들을 응용한 케이스들 소개, Self-Consistency, Generated Knowledge, Least-to-Most, Prompt Chaining, ReAct

### 4-1. Zero-Shot Prompting

 - Zero-Shot이란 추가적인 학습이나 예시/시연 없이 바로 답변 출력을 유도하는 법
    - ChatGPT 전에는 어떤 태스크 진행을 위해서는 데이터 학습이 필요했음
    - 그러나 ChatGPT 3.5, GPT-4, Claude 3 같은 LLM은 추가 학습이나 예시가 없어도 어느 정도 잘 답변하는 편
    - 2022년 ChatGPT 출시 이후 2024년 기준 ChatGPT한테 바로 질문하는 것은 자연스러운 행위

### 4-2. Few-Shot

 - 참고 할 수 있는 정답 사례들을 Prompt에 추가하여 성능을 높이는 방법
 - Language Models are Few-Shot Learners 논문 (NeurIPS 2020, OpenAI)
    - 논문: https://arxiv.org/abs/2005.14165 (=GPT-3 논문)
 - Few Shot 의미
    - 5-shot의 경우 참고 할 정답 사례들을 Prompt에 5개를 입력해줬다는 뜻
    - LLM 평가지표 보면 MMLU(5-shot) 이렇게 적혀있는게 바로 Few-Shot을 적용했다는 뜻
    - 평가에서도 사용될만큼 공인된 Prompt Engineering 방법론
        - OpenAI에서 GPT-4 벤치마크 할 때 모든 Prompt에 Few-Shot 적용했음

```python
# Few-Shot 예시: LLaMA 논문 Figure 3 (https://arxiv.org/pdf/2302.13971.pdf)
# API 키 설정
from google.colab import userdata

OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')

# 클라이언트 생성
from openai import OpenAI

client = OpenAI(
    api_key=OPENAI_API_KEY
)

# Zero-Shot
prompt = """Q: Who wrote the book the origin of species?
"""
completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)


# One-Shot
prompt = """Answer these questions:
Q: Who sang who wants to be a millionaire in high society?
A: Frank Sinatra
Q: Who wrote the book the origin of species?
A: """
completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)
```

### 4-3. Chain-of-Thought

 - Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (NeurIPS 2022, Google)
    - 논문: https://arxiv.org/abs/2201.11903
 - Few Shot이 참고 할 수 있는 정답 사례들을 Prompt에 추가하여 성능을 높이는 방법이라면 Chain of Thought는 거기에 추가로 문제 해결 과정도 같이 Prompt에 추가하는 방식
 - 대부분의 Prompt Engineering 기법은 Chain-of-Thought의 후속

```python
# Prompt 출처: https://github.com/microsoft/generative-ai-for-beginners/tree/main/05-advanced-prompts
# Zero-Shot
prompt = """Alice has 5 apples, throws 3 apples, gives 2 to Bob and Bob gives one back, how many apples does Alice have?"""
completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)
# Alice would have 5 apples - 3 thrown + 2 given to Bob - 1 given back = 3 apples. 
# Therefore, Alice would have 3 apples.

# Chain-of-Thought
prompt = """Lisa has 7 apples, throws 1 apple, gives 4 apples to Bart and Bart gives one back:
7 - 1 = 6
6 - 4 = 2
2 + 1 = 3

Alice has 5 apples, throws 3 apples, gives 2 to Bob and Bob gives one back, how many apples does Alice have?"""
completion = client.chat.completions.create(
    model="gpt-3.5-turbo-0125",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)
# 5 - 3 = 2
# 2 - 2 = 0
# 0 + 1 = 1
# Alice has 1 apple left.

# Prompt 출처: https://github.com/microsoft/generative-ai-for-beginners/tree/main/05-advanced-prompts
prompt = """Alice has 5 apples, throws 3 apples, gives 2 to Bob and Bob gives one back, how many apples does Alice have?"""
completion = client.chat.completions.create(
    model="gpt-4-turbo-2024-04-09",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)
# 1. Alice starts with 5 apples.
# 2. Alice throws 3 apples away. Now she has 5 - 3 = 2 apples.
# 3. Alice gives 2 apples to Bob. Now she has 2 - 2 = 0 apples.
# 4. Bob gives 1 apple back to Alice. Now she has 0 + 1 = 1 apple.
# Therefore, Alice has 1 apple left.
```

## 5. 프롬프트 고도화

 - 평가 기준 설정 > 테스크 케이스 설정(KMMLU 객관식 질문 1개, 간단한 요약 질문 1개) > Baseline Prompt 작성 및 평가 > 고도화

```python
from google.colab import userdata
from openai import OpenAI

OPENAI_API_KEY = userdata.get('OPENAI_API_KEY')
client = OpenAI(api_key=OPENAI_API_KEY)
```

 - `KMMLU 한글 객관식 문제`
```python
# 정답은 B=6
question = 'x, y가 세 부등식 y ≤ x+3, y ≤ -4x+3, y ≥ 0을 만족할 때, x+y의 최댓값을 M, 최솟값을 m이라 하면 M-m의 값은?'
A = 4
B = 6
C = 8
D = 10

# Prompt 출처: KMMLU 논문에서 실제로 평가에 사용한 Prompt (논문: https://arxiv.org/pdf/2402.11548.pdf)
prompt = f"""{question}
A. {A}
B. {B}
C. {C}
D. {D}
정답："""
completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)

# Prompt 출처: KMMLU 논문에서 실제로 평가에 사용한 Prompt (논문: https://arxiv.org/pdf/2402.11548.pdf)
prompt = f"""{question}
A. {A}
B. {B}
C. {C}
D. {D}
정답："""
completion = client.chat.completions.create(
    model='gpt-4-turbo-2024-04-09',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)
```

 - `고도화 포인트 1`
    - Prompt를 한글에서 영어로 수정
    - 페르소나 부여
```python
# Prompt 출처: KMMLU 논문에서 실제로 평가에 사용한 Prompt (논문: https://arxiv.org/pdf/2402.11548.pdf)
prompt = f"""You are an Professional in Mathematics. Below is given a math question in Korean.

{question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer："""

completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)

print(completion.choices[0].message.content)
```

 - `고도화 포인트 2`
    - 차근차근 생각하라고 이야기해주기
```python
# Prompt 출처: KMMLU 논문에서 실제로 평가에 사용한 Prompt (논문: https://arxiv.org/pdf/2402.11548.pdf)
prompt = f"""You are a Professional in Mathematics. Below is given a math question in Korean.
You are to think carefully about the question and choose one of four given answers. Only one of them is true.

{question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer："""

completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)

print(completion.choices[0].message.content)


# 실제 Reasoning을 통해 맞춘건지 확인
# Prompt 출처: KMMLU 논문에서 실제로 평가에 사용한 Prompt (논문: https://arxiv.org/pdf/2402.11548.pdf)
prompt = f"""You are a Professional in Mathematics. Below is given a math question in Korean.
You are to think carefully about the question and choose one of four given answers. Only one of them is true.
Give reasons about why you think your answer is correct.

{question}
A. {A}
B. {B}
C. {C}
D. {D}
Answer："""

completion = client.chat.completions.create(
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)

print(completion.choices[0].message.content)
```
