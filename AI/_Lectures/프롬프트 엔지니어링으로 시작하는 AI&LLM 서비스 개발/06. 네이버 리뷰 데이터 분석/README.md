# 네이버 리뷰 데이터 분석

 - 깃허브: https://github.com/HanSeokhyeon/review_analysis
 - 네이버 리뷰 데이터: https://github.com/e9t/nsmc

## 1. 리뷰 데이터 이해하기

 - id, document, label 3개의 필드가 가진 리뷰 데이터
```python
import pandas as pd
import ssl

# 인증서 오류시
ssl._create_default_https_context = ssl._create_unverified_context

def main():
    url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
    df = pd.read_csv(url, sep='\t')
    print(df.head())

if __name__ == '__main__':
    main()
```

## 2. 리뷰 긍정/부정 평가

 - 태스크 요구사항
    - 주어진 리뷰가 긍정이면 1, 부정이면 0으로 평가
    - API 매개변수로 response_format을 json_object 옵션을 주어 JSON 응답을 받아 파싱 처리
```python
# prompt_template.py
prompt_template = """다음은 영화에 대한 리뷰입니다. 영화에 대해 긍정적이면 1, 부정적이면 0으로 평가해주세요.

'''review
{review}
'''
"""

prompt_template_json = """다음은 영화에 대한 리뷰입니다. 영화에 대해 긍정적이면 1, 부정적이면 0으로 평가해주세요.

아래 json 양식처럼 답변해주세요.
{{
    "score": 0 or 1
}}

'''review
{review}
'''
"""


# inference.py
import json
from openai import Client
from prompt_template import *

client = Client()

# 리뷰가 긍정이면 1, 부정이면 0 반환
def inference(review):
    prompt = prompt_template.format(review=review)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    output = response.choices[0].message.content
    return output

# score 필드에 1 또는 0 반환
def inference_json(review):
    prompt = prompt_template_json.format(review=review)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json

# 요청 토큰, 응답 토큰에 따른 비용 계산
def calculate_cost(prompt_tokens, completion_tokens):
    return (prompt_tokens / 1000000 * 0.5 + completion_tokens / 1000000 * 1.5) * 1340

# 긍정/부정 평가 및 비용 반환
def inference_json_with_cost(review):
    prompt = prompt_template_json.format(review=review)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    cost = calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json, cost


if __name__ == '__main__':
    output, cost = inference_json_with_cost("보는 내내 시간 가는줄 모르고 정말 재밌게 봤습니다~")
    print(output)
    print(f"{cost:.4f}원")
```

 - `API로 만들기(FastAPI)`
    - 실행 명령어: "uvicorn main:app --reload"
```python
# pip install fastapi
# pip install "uvicorn[standard]"

from fastapi import FastAPI, Body
from pydantic import BaseModel
from inference import inference_json

app = FastAPI()

class RequestBody(BaseModel):
    review: str

@app.post("/evaluate")
async def evaluate_review(body: RequestBody = Body()):
    return inference_json(body.review)
```

 - `데모로 만들기(Streamlit)`
    - 공식 문서: https://docs.streamlit.io/develop/api-reference
```python
# pip install streamlit

# 데모1
import streamlit as st
from inference import inference_json

review = st.text_input('리뷰', '이 영화 재밌어요!')
if st.button('submit'):
    score = inference_json(review)
    st.write(score)

# 데모2
import pandas as pd
import streamlit as st
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

url = "https://raw.githubusercontent.com/e9t/nsmc/master/ratings_test.txt"
df = pd.read_csv(url, sep='\t')
review = st.selectbox('리뷰', df.iloc[:10]['document'])
if st.button('submit'):
    score = inference_json(review)
    st.write(score)
```

## 3. 리뷰 중요 키워드 추출(긍정 키워드, 부정 키워드 추출)

 - `prompt_template.py`
```python
prompt_template = """다음은 영화에 대한 리뷰입니다. 리뷰에서 긍정적인 키워드, 부정적인 키워드를 추출해주세요.

'''review
{review}
'''
"""

prompt_template_function_calling = """다음은 영화에 대한 리뷰입니다. 리뷰에서 긍정적인 키워드, 부정적인 키워드를 추출해주세요.
JSON으로 응답해주세요.

'''review
{review}
'''
"""
```

 - `inference.py`
    - Function Calling이란 GTP가 사용할 function(tool)을 미리 지정하고 상황에 따라 적절히 function을 사용하는 기능
    - Function Calling의 arguments 양식을 지정하기 위해서 json schema를 지정해주어야 한다.
    - 주의사항: Function Calling을 사용하면 Tool 없이 GPT를 사용하는 것보다 다소 성능이 낮음
        - 모델 업그레이드(GPT 3.5 -> GPT 4)
        - 프롬프트 개선(한글 > 영어)
        - Function Calling 대신 JSON 응답 사용
        - Langchin 자동 생성 프롬프트 사용
```python
import json
from openai import Client
from prompt_template import prompt_template, prompt_template_function_calling

client = Client()

def inference(review):
    prompt = prompt_template.format(review=review)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    output = response.choices[0].message.content
    return output

def inference_function_calling(review):
    prompt = prompt_template_function_calling.format(review=review)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "extract_positive_and_negetive_keywords",
                "description": "Extract positive keywords and negative keywords in given movie review.",
                "parameters": {
                    "$schema": "http://json-schema.org/draft-07/schema#",
                    "type": "object",
                    "properties": {
                        "positive_keywords": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        },
                        "negative_keywords": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            }
                        }
                    },
                    "required": ["positive_keywords", "negative_keywords"]
                }
            }
        }
    ]
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"},
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "extract_positive_and_negetive_keywords"}}
    )
    output = response.choices[0].message.tool_calls[0].function.arguments
    output_json = json.loads(output)
    return output_json


if __name__ == '__main__':
    # print(inference_function_calling("보는 내내 시간 가는줄 모르고 정말 재밌게 봤습니다~"))
    print(inference_function_calling("정말 쓰레기같은 영화... 다신 안본다"))
```

## 4. 리뷰 요약 및 분석

 - `prompt_template.py`
```python
prompt_template = """다음은 영화에 대한 리뷰들입니다. 리뷰 내용을 종합적으로 요약해주세요.
Json으로 응답해주세요.

'''reviews
{reviews}
'''"""

prompt_template_langchain = """다음은 영화에 대한 리뷰들입니다. 리뷰 내용을 종합적으로 요약해주세요.

{format_instructions}

'''```'''reviews
{reviews}
'''```'''

Answer in the following language: Korean
"""
```

 - `inference.py`
    - LangChain은 LLM 모델을 사용하는 파이프라인을 구성하는 프레임워크. 프롬프트 생성 등 자동으로 처리해주는 기능들이 편리하다.
    - PromptTemplate: 프롬프트를 자동으로 생성해주는 모듈
        - FewShotPromptTemplate
        - SystemMessagePromptTemplate
        - UserMessagePromptTemplate
    - ChatOpenAI: OpenAI 모델을 사용할 수 있는 모듈
        - Anthropic, Google Gemini
        - LLaMA3 같은 Local 모델을 띄워서 사용도 가능
    - OutputParser: LLM의 출력을 적절한 구조로 parsing하는 모듈
        - StrOutputParser
        - JsonOutputParser
        - YamlOutputParser
```python
# pip install langchain
# pip install langchain-openai
import json
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import Client
from pydantic import BaseModel
from prompt_template import prompt_template, prompt_template_langchain

client = Client()

def inference(reviews):
    reviews = "\n".join(reviews)
    prompt = prompt_template.format(reviews=reviews)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )
    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json


class Output(BaseModel):
    summary: str


output_parser = PydanticOutputParser(pydantic_object=Output)
prompt_maker = PromptTemplate(
    template=prompt_template_langchain,
    input_variables=["reviews"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)
model = ChatOpenAI(
    temperature=0.0,
    openai_api_key="",
    model_name="gpt-3.5-turbo"
)
chain = (prompt_maker | model | output_parser)


def inference_langchain(reviews):
    reviews = "\n".join(reviews)
    # prompt = prompt_maker.invoke({"reviews": reviews})
    output = chain.invoke({"reviews": reviews})
    return output.summary


if __name__ == '__main__':
    print(inference_langchain([
        "정말 재미없네요.",
        "시간 가는줄 모르고 정말 즐겁게 봤습니다.",
        "로다쥬 나오는 영화는 무조건 추천이죠",
        "다음 편 너무 기대...."
    ]))
```
