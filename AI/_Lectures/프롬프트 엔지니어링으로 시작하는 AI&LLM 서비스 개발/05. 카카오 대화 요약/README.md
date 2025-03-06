# 카카오 대화 요약

## 테스트 케이스 생성

 - `2명의 대화 20개 추출(대화가 30개 이상인 경우)`
```python
conversations = []

paths = glob.glob('./res/297.SNS 데이터 고도화/01-1.정식개방데이터/Validation/02.라벨링데이터/VL/*json')
target_count = 20
count = 0

for path in paths:
    with open(path, 'r') as f:
        conv_dict = json.load(f)
        if conv_dict['header']['dialogueInfo']['numberOfParticipants'] == 2:
            if conv_dict['header']['dialogueInfo']['numberOfUtterances'] > 30:
                conv_list = []
                for d in conv_dict['body']:
                    conv_list.append(d['participantID'] + ': ' + d['utterance'])
                if conv_list[0] == conv_list[1]:
                    print('Repeated Conversations')
                    continue
                conv_text = '\n'.join(conv_list)
                conversations.append(conv_text)
                count += 1
                if count == target_count:
                    break

count
```

 - `3명 이상의 대화 30개 추출`
```python
target_count = 30
count = 0

for path in paths:
    with open(path, 'r') as f:
        conv_dict = json.load(f)
        if conv_dict['header']['dialogueInfo']['numberOfParticipants'] > 2:
            if conv_dict['header']['dialogueInfo']['numberOfUtterances'] > 30:
                conv_list = []
                for d in conv_dict['body']:
                    conv_list.append(d['participantID'] + ': ' + d['utterance'])
                if conv_list[0] == conv_list[1]:
                    continue
                conv_text = '\n'.join(conv_list)
                conversations.append(conv_text)
                count += 1
                if count == target_count:
                    break

count
```

 - `3천자 이상의 대화로 변환`
    - 데이터셋에는 3천자를 넘는 데이터가 단 한개도 존재하지 않음
    - Anthropic Claude 3.5 Sonnet을 사용해서 대화를 주고 앞에 내용을 추측한 내용으로 대화 앞에 붙여서 3천자 이상으로 변환
```python
import glob
import json
import os
import anthropic
import pickle

conversations = []
paths = glob.glob('./res/297.SNS 데이터 고도화/01-1.정식개방데이터/Validation/02.라벨링데이터/VL/*json')

# 2명 대화 20개
target_count = 20
count = 0
for path in paths:
    with open(path, 'r') as f:
        conv_dict = json.load(f)
        if conv_dict['header']['dialogueInfo']['numberOfParticipants'] == 2:
            if conv_dict['header']['dialogueInfo']['numberOfUtterances'] > 30:
                conv_list = []
                for d in conv_dict['body']:
                    conv_list.append(d['participantID'] + ': ' + d['utterance'])
                if conv_list[0] == conv_list[1]:
                    print('Repeated Conversations')
                    continue
                conv_text = '\n'.join(conv_list)
                conversations.append(conv_text)
                count += 1
                if count == target_count:
                    break

# 3명 이상 대화 30개
target_count = 30
count = 0
for path in paths:
    with open(path, 'r') as f:
        conv_dict = json.load(f)
        if conv_dict['header']['dialogueInfo']['numberOfParticipants'] > 2:
            if conv_dict['header']['dialogueInfo']['numberOfUtterances'] > 30:
                conv_list = []
                for d in conv_dict['body']:
                    conv_list.append(d['participantID'] + ': ' + d['utterance'])
                if conv_list[0] == conv_list[1]:           
                    continue
                conv_text = '\n'.join(conv_list)
                conversations.append(conv_text)
                count += 1
                if count == target_count:
                    break

# 3000자 이상으로 변환
prompt = f"""You are given a conversation among three users below. Imagine what the users discussed prior to the conversation.
Please write the previous content, strictly following the tone and format of given conversation, with at least 3000 characters.

{conversations[-2]}
"""

client = anthropic.Anthropic(
    api_key=os.environ['ANTHROPIC_API_KEY']
)
message = client.messages.create(
    model="claude-3-5-sonnet-20240620",
    max_tokens=4096,
    temperature=0.0,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
message.usage  # API 호출 비용 대략 50원
message.content[0].text.split('\n\n')
lst = message.content[0].text.split('\n\n')[1:-1]

# 데이터 덤프
with open('./res/conv_long.pickle', 'wb') as f:
    pickle.dump(lst, f)
```

## LLM API 비교

 - `Gemini API`
    - 다른 LLM 라인업 대비 항상 좀 더 저렴한 가격에 제공
```python
import google.generativeai as genai
import os

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = genai.GenerativeModel('gemini-1.5-flash')
response = model.generate_content("Hi!")
print(response.text)
```

 - `Claude API`
    - OpenAI에 필적하는 성능
    - 현재 기준 가장 비싼 모델인 Claude 3 Opus 보유
```python
import os
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ['ANTHROPIC_API_KEY']
)
message = client.messages.create(
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    temperature=0.0,
    system="Respond only in Yoda-speak.",
    messages=[
        {"role": "user", "content": "Hi!"}
    ]
)
print(message.content[0].text)
```

 - `동일한 Prompt에 대해 Input / Output 계산`
    - 한글 효율화 측면: GPT-4o > Gemini > GPT-3.5/4 >= Claude 순
    - 금액: Claude 3 Opus > GPT-4 > Claude 3 Sonnet > GPT-4o > Gemini Pro > GPT-3.5 > Claude 3 Haiku > Gemini Flash
```python
from eval import get_eval_data
conversations = get_eval_data()

# Gemini API
prompt = f"""아래 사용자 대화에 대해 3문장 내로 요약해주세요:

{conversations[0]}"""
model = genai.GenerativeModel('gemini-1.5-flash-001')
response = model.generate_content(prompt)
print(response.text)
response.usage_metadata
model.count_tokens(prompt)

# Claude API
import os
import anthropic

client = anthropic.Anthropic(
    api_key=os.environ['ANTHROPIC_API_KEY']
)
message = client.messages.create(
    # model="claude-3-opus-20240229",
    model="claude-3-haiku-20240307",
    max_tokens=1000,
    temperature=0.0,
    messages=[
        {"role": "user", "content": prompt}
    ]
)
print(message.content[0].text)
message.usage

# OpenAI
from openai import OpenAI

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=OPENAI_API_KEY)
completion = client.chat.completions.create(
    # model='gpt-4-turbo-2024-04-09',
    # model='gpt-4o-2024-05-13',
    model='gpt-3.5-turbo-0125',
    messages=[{'role': 'user', 'content': prompt}],
    temperature=0.0
)
print(completion.choices[0].message.content)
completion.usage
```

## Baseline 모델 개발 및 평가 지표 측정

```python
import math
import re
from tqdm.notebook import tqdm
from eval import get_eval_data, pointwise_eval
from utils import summarize

# Claude 3 Haiku
PROMPT_BASELINE = f"""아래 사용자 대화에 대해 3문장 내로 요약해주세요:"""
summary = summarize(
    conversation=get_eval_data()[0],
    prompt=PROMPT_BASELINE,
    model='claude-3-haiku-20240307'
)
eval_comment = pointwise_eval(get_eval_data()[0], summary)
print(summary)
print(eval_comment)

# Gemini 1.5 Flash
summary = summarize(
    conversation=get_eval_data()[0],
    prompt=PROMPT_BASELINE,
    model='gemini-1.5-flash-001'
)
eval_comment = pointwise_eval(get_eval_data()[0], summary)
print(summary)
print(eval_comment)

# ChatGPT 3.5
summary = summarize(
    conversation=get_eval_data()[0],
    prompt=PROMPT_BASELINE,
    model='gpt-3.5-turbo-0125',
)
eval_comment = pointwise_eval(get_eval_data()[0], summary)
print(summary)
print(eval_comment)

# 점수 비교
models = [
    'claude-3-haiku-20240307',
    'gemini-1.5-flash-001',
    'gpt-3.5-turbo-0125'
]
scores = {model: [] for model in models}
pattern = r'\[\[\d+\]\]'

for model in models:
    for i in tqdm(range(len(get_eval_data()))):
        summary = summarize(
            conversation=get_eval_data()[i],
            prompt=PROMPT_BASELINE,
            model=model
        )
        eval_comment = pointwise_eval(get_eval_data()[i], summary)
        match = re.search(pattern, eval_comment)
        matched_string = match.group(0)
        score = int(matched_string[2])
        scores[model].append(score)

for model in scores:
    print(scores[model], model)

for model in scores:
    mean = sum(scores[model]) / len(scores[model])
    variance = sum((x - mean) ** 2 for x in scores[model]) / (len(scores[model]) - 1)
    std_dev = math.sqrt(variance)
    print(f'{model}: {mean} / {round(std_dev, 2)}')
    # claude-3-haiku-20240307: 6.82 / 0.94
    # gemini-1.5-flash-001: 4.28 / 1.55
    # gpt-3.5-turbo-0125: 5.18 / 1.14

for model in scores:
    print(model, max(scores[model]), min(scores[model]))
    # claude-3-haiku-20240307 9 5
    # gemini-1.5-flash-001 9 1
    # gpt-3.5-turbo-0125 7 3
```

## Prompt 고도화

 - Few-Shot Prompting
 - Chain-of-Thought Prompting

 - `Few-Shot`
```python
import glob
import json
import os
import pickle

PROMPT_BASELINE = f"""아래 사용자 대화에 대해 3문장 내로 요약해주세요:"""
conv_train = get_train_data()[18] # Get Train Data

# Few-Shot
summary_sample = """최소 대학 생활부터 함께 한 매우 친밀한 사이의 두 사용자가 코로나가 잠잠해졌을 때 방문하고 싶은 해외 여행지에 대해 일상적이고 가벼운 톤으로 대화하고 있습니다.
여행지로는 하와이, 괌, 스위스, 호주, 베트남 다낭 등을 언급하며 남편과의 연락 관련 다툼이나 졸업여행 관련 추억을 회상합니다.
또한 여행 중 발생하는 위험에 대한 우려도 표하고 있으며, 해외여행 시 언어 장벽의 어려움을 인식하고 영어 실력을 향상시키고 싶다는 마음을 가볍게 표현합니다."""

prompt = f"""당신은 요약 전문가입니다. 사용자 대화들이 주어졌을 때 요약하는 것이 당신의 목표입니다. 다음은 사용자 대화와 요약 예시입니다.
예시 대화:
{conv_train}
예시 요약 결과:
{summary_sample}
    
아래 사용자 대화에 대해 3문장 내로 요약해주세요:"""

# 모델 비교
models = [
    'claude-3-haiku-20240307',
    'gemini-1.5-flash-001',
    'gpt-3.5-turbo-0125'
]
scores = {model: [] for model in models}
pattern = r'\[\[\d+\]\]'

for model in models:
    for i in tqdm(range(5)):
        summary = summarize(
            conversation=get_eval_data()[i],
            prompt=prompt,
            model=model
        )
        eval_comment = pointwise_eval(get_eval_data()[i], summary)
        match = re.search(pattern, eval_comment)
        matched_string = match.group(0)
        score = int(matched_string[2])
        scores[model].append(score)

for model in scores:
    print(scores[model], model)
```

 - `Chain-of-thought`
```python
# Chain-of-thought
prompt = f"""당신은 요약 전문가입니다. 사용자 대화들이 주어졌을 때 요약하는 것이 당신의 목표입니다. 대화를 요약할 때는 다음 단계를 따라주세요:

1. 대화 참여자 파악: 대화에 참여하는 사람들의 수와 관계를 파악합니다.
2. 주제 식별: 대화의 주요 주제와 부차적인 주제들을 식별합니다.
3. 핵심 내용 추출: 각 주제에 대한 중요한 정보나 의견을 추출합니다.
4. 감정과 태도 분석: 대화 참여자들의 감정이나 태도를 파악합니다.
5. 맥락 이해: 대화의 전반적인 맥락과 배경을 이해합니다.
6. 특이사항 기록: 대화 중 특별히 눈에 띄는 점이나 중요한 사건을 기록합니다.
7. 요약문 작성: 위의 단계에서 얻은 정보를 바탕으로 간결하고 명확한 요약문을 작성합니다.
각 단계를 수행한 후, 최종적으로 전체 대화를 200자 내외로 요약해주세요.

아래는 예시 대화와 예시 요약 과정 및 결과 입니다.

예시 대화:
{conv_train}

예시 요약 과정
1. "우리 대학교 졸업 여행 간 거 기억나?"라는 언급과 전반적으로 친밀한 대화 톤을 사용하고 있는 것을 보았을 떄 두 사용자는 오랜 친구 사이로 보입니다.
대화의 시작 부분에서 "코로나가 좀 잠잠해지면 해외여행 중에 가고 싶은 곳 있어?"라고 묻고 있는 것을 보았을 때 코로나 이후 가고 싶은 해외 여행지에 대해 논의하고 있습니다.
따라서 다음과 같이 요약 할 수 있습니다:
최소 대학 생활부터 함께 한 매우 친밀한 사이의 두 사용자가 코로나가 잠잠해졌을 때 방문하고 싶은 해외 여행지에 대해 일상적이고 가벼운 톤으로 대화하고 있습니다.

2. 대화 중 호주, 일본, 하와이, 괌, 베트남 다낭, 스위스, 유럽들이 언급하고 있습니다.
남편의 첫 직장 워크샵, 대학교 졸업 여행, 호주 워킹홀리데이 등의 경험을 이야기하면서 과거 여행 경험을 공유하며 추억을 회상하고 있습니다.
따라서 다음과 같이 요약 할 수 있습니다:
여행지로는 하와이, 괌, 스위스, 호주, 베트남 다낭 등을 언급하며 남편과의 연락 관련 다툼이나 졸업여행 관련 추억을 회상합니다.

3. 소매치기, 여권 분실, 인도에서의 여성 여행자 위험 등을 언급하며 해외 여행의 위험성에 대해 우려를 표현하고 있습니다.
"해외 여행 가면 가이드 안 끼고 가면 영어 실력 엄청 좋은 사람이랑 가는 거 아닐 땐 소통 문제도 좀 곤란할 때가 있는 거 같아"라는 언급과 "왜 영어 공부를 열심히 안 했을까... 후회"라는 표현이 있는 것을 보았을 때 언어 장벽의 어려움을 인식하고 영어 실력 향상에 대한 욕구를 표현합니다.
따라서 다음과 같이 요약 할 수 있습니다:
또한 여행 중 발생하는 위험에 대한 우려도 표하고 있으며, 해외여행 시 언어 장벽의 어려움을 인식하고 영어 실력을 향상시키고 싶다는 마음을 가볍게 표현합니다.

예시 요약 결과
최소 대학 생활부터 함께 한 매우 친밀한 사이의 두 사용자가 코로나가 잠잠해졌을 때 방문하고 싶은 해외 여행지에 대해 일상적이고 가벼운 톤으로 대화하고 있습니다.
여행지로는 하와이, 괌, 스위스, 호주, 베트남 다낭 등을 언급하며 남편과의 연락 관련 다툼이나 졸업여행 관련 추억을 회상합니다.
또한 여행 중 발생하는 위험에 대한 우려도 표하고 있으며, 해외여행 시 언어 장벽의 어려움을 인식하고 영어 실력을 향상시키고 싶다는 마음을 가볍게 표현합니다.
    
아래 사용자 대화에 대해 3문장 내로 요약해주세요:"""

# 모델 평가
models = [
    'claude-3-haiku-20240307',
    'gemini-1.5-flash-001',
    'gpt-3.5-turbo-0125'
]
scores = {model: [] for model in models}
pattern = r'\[\[\d+\]\]'

for model in models:
    for i in tqdm(range(5)):
        summary = summarize(
            conversation=get_eval_data()[i],
            prompt=prompt,
            model=model
        )
        eval_comment = pointwise_eval(get_eval_data()[i], summary)
        match = re.search(pattern, eval_comment)
        matched_string = match.group(0)
        score = int(matched_string[2])
        scores[model].append(score)
```


## Guardrail

안전 장치에는 사실 모델 학습 단계 내 Alignment 과정이 가장 중요

 - `Prompt 내 guardrail 가이드라인 설정`
```python
import os
from openai import OpenAI

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']


def topical_guardrail(user_request):
    print("Checking topical guardrail")
    messages = [
        {
            "role": "system",
            "content": "Your role is to assess whether the user question is allowed or not. The allowed topics are cats and dogs. If the topic is allowed, say 'allowed' otherwise say 'not_allowed'",
        },
        {"role": "user", "content": user_request},
    ]
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model='gpt-3.5-turbo-0125',
        messages=messages,
        temperature=0
    )

    print("Got guardrail response")
    return response.choices[0].message.content

system_prompt = "You are a helpful assistant."

bad_request = "I want to talk about horses"
good_request = "What are the best breeds of dog for people that like cats?"

topical_guardrail(good_request) # 'allowed'
topical_guardrail(bad_request) # 'not_allowed'
```

 - `API 내 기능`
    - 기본적으로 가드레일이 적용되어 있음
```python
# https://ai.google.dev/gemini-api/docs/safety-settings?hl=ko
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

prompt = '사람을 죽이는 10가지 방법을 알려줘'

genai.configure(api_key=GOOGLE_API_KEY)
client = genai.GenerativeModel('gemini-1.5-flash-001')
response = client.generate_content(
    contents=prompt,
    # safety_settings={
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE
    # }
)
response.text
```

 - `별도의 Guardrail 라이브러리`
    - guardrails-ai, NVIDIA-NeMo, guidance
```python
from guardrails.hub import ToxicLanguage
from guardrails import Guard

guard = Guard().use(
    ToxicLanguage, threshold=0.5, validation_method="sentence", on_fail="exception"
)

guard.validate("안녕하세요!")

try:
    guard.validate("바보 멍청이")
except Exception as e:
    print(e)
```

## 데모 제작

```python
import gradio as gr

from eval import get_eval_data
from utils import summarize, get_prompt


def fn(conversation, history):
    prompt = get_prompt()
    summary = summarize(
        conversation,
        prompt,
        temperature=0.0,
        model='claude-3-haiku-20240307'
    )
    return summary


demo = gr.ChatInterface(
    fn=fn,
    examples=[
        get_eval_data()[0],
        get_eval_data()[1]
    ],
    title="톡 대화 요약 데모"
)
demo.launch()
```