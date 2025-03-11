from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer


def get_data():
    url = "https://fastcampus.co.kr/data_online_llmservice"

    # HTML 문서 로드(SSL 인증 비활성화)
    loader = AsyncHtmlLoader(url, verify_ssl=False)
    docs = loader.load()

    # HTML 문서를 텍스트로 변환
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)

    # 변환된 문서 리스트에서 첫 번쨰 문서의 텍스트 콘텐츠 추출
    content = docs_transformed[0].page_content
    return content


if __name__ == '__main__':
    print(get_data())