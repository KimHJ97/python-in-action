from fastapi import FastAPI, Body
from pydantic import BaseModel

from search_and_answer import search, generate_answer

app = FastAPI()


class RequestBody(BaseModel):
    question: str


@app.post("/answer")
async def answer(body: RequestBody = Body()):
    """
    API 엔드포인트: /answer (POST 요청)
    - 입력: JSON 형식의 질문 (RequestBody 모델 사용)
    - 처리: FAISS 벡터 DB에서 질문 검색 후, OpenAI 모델을 사용하여 답변 생성
    - 출력: 생성된 답변을 반환
    """
    qa = search(body.question)
    return generate_answer(qa, body.question)