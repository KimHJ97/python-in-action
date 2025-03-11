import json
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from openai import Client
from pydantic import BaseModel

from ch2.download_data import get_urls
from ch2.prompt_template import prompt_template

client = Client(api_key="...")


# 하나의 이미지 URL을 OpenAI 모델에 입력하여 FAQ 생성을 요청하는 함수
def inference(url_list):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "다음 사진의 내용을 읽고 FAQ를 한국어로 만들어주세요."},
                    {"type": "image_url",
                     "image_url": {
                         "url": url_list[0]
                     }}
                ]
            }
        ],
        max_tokens=1000
    )
    output = response.choices[0].message.content
    return output


# 여러 개(최대 5개)의 이미지 URL을 OpenAI 모델에 입력하여 FAQ 생성을 요청하는 함수
def inference_many(url_list):
    content = [
        {"type": "text", "text": "다음 사진들의 내용을 읽고 FAQ를 한국어로 만들어주세요."}
    ]

    # 최대 5개의 이미지 URL을 추가
    for url in url_list[:5]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=1000
    )

    # 응답에서 생성된 FAQ 내용 추출
    output = response.choices[0].message.content
    return output

# 질문(QA) 데이터를 표현하는 Pydantic 모델
class QA(BaseModel):
    question: str
    answer: str

# 최종 출력 데이터 모델
class Output(BaseModel):
    qa_list: List[QA]

# Pydantic을 사용하여 응답을 구조화하는 파서 생성
output_parser = PydanticOutputParser(pydantic_object=Output)

# 여러 개(최대 5개)의 이미지 URL을 OpenAI 모델에 입력하여 JSON 형식으로 FAQ를 생성하는 함수
def inference_many_json(url_list):
    # JSON 출력 형식에 맞춘 프롬프트 생성
    prompt = prompt_template.format(format_instructions=output_parser.get_format_instructions())

    content = [
        {"type": "text", "text": prompt}
    ]

    # 최대 5개의 이미지 URL을 추가
    for url in url_list[:5]:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": url
            }
        })

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_tokens=1000,
        response_format={"type": "json_object"}
    )

    # 응답에서 JSON 데이터 추출 및 파싱
    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json


if __name__ == '__main__':
    url_list = get_urls()
    result = inference_many_json(url_list)
    print(json.dumps(result, indent=2, ensure_ascii=False))