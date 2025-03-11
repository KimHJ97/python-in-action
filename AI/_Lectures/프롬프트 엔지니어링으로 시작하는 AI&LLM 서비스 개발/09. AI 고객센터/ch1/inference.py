import json
from typing import List

from langchain_core.output_parsers import PydanticOutputParser
from openai import Client
from pydantic import BaseModel

from ch1.download_data import get_data
from ch1.prompt_template import prompt_template, prompt_template_json

client = Client(api_key="...")


def inference(product_detail):
    # 주어진 상품 상세 정보를 기반으로 프롬프트 생성
    prompt = prompt_template.format(product_detail=product_detail)

    # OpenAI GPT 모델에 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # 응답에서 생성된 텍스트 추출
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


def calculate_cost(prompt_tokens, completion_tokens):
    """
    토큰 수(prompt_tokens, completion_tokens)를 바탕으로 비용을 계산하는 함수
    - prompt_tokens: 프롬프트에 사용된 토큰 수
    - completion_tokens: 응답 생성에 사용된 토큰 수
    - OpenAI 요금 기준:
      - 프롬프트: 100만 토큰당 0.15 USD
      - 응답: 100만 토큰당 0.6 USD
    - 환율(1380원/USD)을 적용하여 비용 반환
    """
    return (prompt_tokens / 1000000 * 0.15 + completion_tokens / 1000000 * 0.6) * 1380


def inference_json(product_detail):
    # JSON 출력 형식에 맞춘 프롬프트 생성
    prompt = prompt_template_json.format(
        format_instructions=output_parser.get_format_instructions(),
        product_detail=product_detail
    )

    # OpenAI GPT 모델에 JSON 형식으로 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    # 요청에 사용된 토큰 수를 바탕으로 비용 계산
    cost = calculate_cost(response.usage.prompt_tokens, response.usage.completion_tokens)
    print(cost)

    # 응답에서 JSON 데이터 추출 및 파싱
    output = response.choices[0].message.content
    output_json = json.loads(output)
    return output_json


if __name__ == '__main__':
    product_detail = get_data()
    result = inference_json(product_detail)
    print(json.dumps(result, indent=2, ensure_ascii=False))