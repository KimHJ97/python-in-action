import pickle

import pandas as pd
import streamlit as st

from search_and_answer import search, generate_answer

# 저장된 FAQ 데이터를 불러오기
with open("qas.pkl", "rb") as f:
    qas = pickle.load(f)

# 불러온 FAQ 데이터를 Pandas DataFrame으로 변환
df = pd.DataFrame(qas)

# Streamlit을 사용하여 데이터프레임을 웹 UI에 표시
st.dataframe(df)

# 사용자 입력 필드 (질문 입력)
question = st.text_input("Question", "이벤트 언제까지에요?")

# 'Submit' 버튼 클릭 시 실행되는 로직
if st.button("Submit"):
    # FAISS 벡터 DB에서 관련된 질문 검색
    qa = search(question)

    # 검색된 질문을 기반으로 OpenAI 모델을 이용해 추가 답변 생성
    answer = generate_answer(qa, question)

    # 생성된 답변을 Streamlit UI에 출력
    st.write(answer)