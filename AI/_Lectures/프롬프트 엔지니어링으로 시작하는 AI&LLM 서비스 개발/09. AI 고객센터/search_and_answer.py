# OpenAI 임베딩 및 벡터 데이터베이스(FAISS) 관련 라이브러리
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from openai import Client

from prompt_template import prompt_template

api_key = "..."
client = Client(api_key="...")


def search(question):
    """
    주어진 질문을 기반으로 FAISS 벡터 DB에서 가장 유사한 질문을 검색하여 관련 FAQ를 반환
    """
    # 저장된 FAISS 인덱스를 로드 (위험한 직렬화 허용)
    db = FAISS.load_local(
        "qas.index",
        OpenAIEmbeddings(openai_api_key=api_key),
        allow_dangerous_deserialization=True
    )

    # 주어진 질문과 가장 유사한 문서를 검색 (similarity 기반)
    result = db.search(question, search_type="similarity")

    # 가장 유사한 FAQ의 메타데이터(원본 질문 및 답변)를 반환
    return result[0].metadata


def generate_answer(context, question):
    """
    검색된 FAQ 데이터를 활용하여 사용자의 질문에 대한 응답을 생성
    """
    # 기존 FAQ 데이터를 문자열로 구성 (FAQ를 기반으로 GPT 모델이 답변할 수 있도록 함)
    context_join = f"""Q: {context['question']}
A: {context['answer']}"""
    
    # FAQ 기반으로 새로운 질문에 대한 답변을 생성하기 위한 프롬프트 구성
    prompt = prompt_template.format(context=context_join, question=question)

    # OpenAI 모델을 사용하여 답변 생성 요청
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    # 모델이 생성한 답변 반환
    output = response.choices[0].message.content
    return output


if __name__ == '__main__':
    question = "실무에서 LLM 기반 서비스 개발할 때 문제점?"
    qa = search(question)

    # 검색된 FAQ 출력
    print(qa['question'])
    print(qa['answer'])
    print()

    # 검색된 FAQ를 기반으로 추가적인 답변 생성 및 출력
    print(question)
    print(generate_answer(qa, question))