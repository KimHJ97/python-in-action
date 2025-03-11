import pickle

# LangChain에서 제공하는 OpenAI 임베딩 및 벡터 저장소(FASS)
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from ch1.download_data import get_data
from ch1.inference import inference_json
from ch2.download_data import get_urls
from ch2.inference import inference_many_json


api_key = "..."


def main():
    """
    1. HTML에서 텍스트 및 이미지 데이터를 추출하여 FAQ를 생성
    2. 생성된 FAQ 데이터를 파일로 저장
    3. 질문 목록을 벡터 DB(FAISS)에 임베딩하여 인덱싱 후 저장
    """
    # 1. HTML에서 텍스트 데이터를 추출하고 FAQ 생성
    # html -> text -> faq
    product_detail = get_data()
    result_text = inference_json(product_detail)

    # 2. HTML에서 이미지 URL을 추출하고 FAQ 생성
    # html -> image -> faq
    url_list = get_urls()
    result_image = inference_many_json(url_list)

    # 3. 텍스트 기반 FAQ와 이미지 기반 FAQ를 합침
    result = result_text["qa_list"] + result_image["qa_list"]
    print(result)

    # 4. FAQ 데이터를 pickle 파일(qas.pkl)로 저장
    with open("qas.pkl", "wb") as f:
        pickle.dump(result, f)

    # 5. FAQ에서 질문(question) 리스트만 추출
    result_questions = [row['question'] for row in result]

    # 6. 벡터 데이터베이스(FAISS) 생성 및 저장
    # vector db indexing
    db = FAISS.from_texts(
        result_questions,
        embedding=OpenAIEmbeddings(openai_api_key=api_key),
        metadatas=result
    )

    # 7. 벡터 DB를 로컬 파일(qas.index)로 저장
    db.save_local("qas.index")


if __name__ == '__main__':
    main()