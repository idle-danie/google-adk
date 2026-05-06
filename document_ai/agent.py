import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.cloud import discoveryengine_v1 as discoveryengine

load_dotenv()

_PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
_SEARCH_LOCATION = os.environ["SEARCH_LOCATION"]
_AGENT_MODEL = os.environ["AGENT_MODEL"]
_DEFAULT_DOCUMENT_GCS_URI = os.environ["DEFAULT_DOCUMENT_GCS_URI"]

_ENGINE_ID = os.environ["ENGINE_ID"]
_DS_LAYOUT = os.environ["DS_LAYOUT_01"]
_DS_OCR = os.environ["DS_OCR_01"]
_DS_DEFAULT = os.environ["DS_DEFAULT_01"]


def _search(query: str, datastore_id: str) -> dict:
    client = discoveryengine.SearchServiceClient()
    serving_config = (
        f"projects/{_PROJECT_ID}/locations/{_SEARCH_LOCATION}/"
        f"collections/default_collection/engines/{_ENGINE_ID}/"
        f"servingConfigs/default_config"
    )
    datastore_path = (
        f"projects/{_PROJECT_ID}/locations/{_SEARCH_LOCATION}/"
        f"collections/default_collection/dataStores/{datastore_id}"
    )
    response = client.search(
        discoveryengine.SearchRequest(
            serving_config=serving_config,
            query=query,
            page_size=5,
            data_store_specs=[
                discoveryengine.SearchRequest.DataStoreSpec(
                    data_store=datastore_path
                )
            ],
            content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
                extractive_content_spec=discoveryengine.SearchRequest.ContentSearchSpec.ExtractiveContentSpec(
                    max_extractive_answer_count=3,
                    max_extractive_segment_count=3,
                ),
                snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                    return_snippet=True,
                    max_snippet_count=5,
                ),
                summary_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec(
                    summary_result_count=5,
                    include_citations=True,
                    model_prompt_spec=discoveryengine.SearchRequest.ContentSearchSpec.SummarySpec.ModelPromptSpec(
                        preamble=(
                            "사용자의 질문에 대해 문서에서 찾은 실제 내용을 상세히 서술하여 답변하십시오. "
                            "'X 페이지를 참고하세요' 또는 '섹션 X에서 다루고 있습니다'처럼 위치만 안내하지 마십시오. "
                            "문서에 있는 구체적인 내용, 절차, 수치, 조건 등을 직접 설명하십시오."
                        )
                    ),
                ),
            ),
        )
    )

    if response.summary and response.summary.summary_text:
        return {"status": "success", "answer": response.summary.summary_text}

    contents = []
    for result in response.results:
        data = result.document.derived_struct_data or {}
        for item in data.get("extractive_answers", []):
            if item.get("content"):
                contents.append(item["content"])
        for item in data.get("extractive_segments", []):
            if item.get("content"):
                contents.append(item["content"])
        for item in data.get("snippets", []):
            if item.get("snippet"):
                contents.append(item["snippet"])

    return {"status": "success", "results": contents}


def search_layout(query: str) -> dict:
    """Layout 파서 데이터스토어에서 검색합니다. 표, 도표, 레이아웃 구조 관련 질문에 적합합니다."""
    return _search(query, _DS_LAYOUT)


def search_ocr(query: str) -> dict:
    """OCR 파서 데이터스토어에서 검색합니다. 정확한 원문 텍스트나 이미지 내 텍스트 검색에 적합합니다."""
    return _search(query, _DS_OCR)


def search_default(query: str) -> dict:
    """Default 파서 데이터스토어에서 검색합니다. 일반적인 문서 내용 검색에 적합합니다."""
    return _search(query, _DS_DEFAULT)


def compare_search(query: str) -> dict:
    """세 파서(Layout, OCR, Default)의 검색 결과를 동시에 반환합니다. 파서별 성능 비교에 사용하세요."""
    return {
        "query": query,
        "layout": _search(query, _DS_LAYOUT),
        "ocr": _search(query, _DS_OCR),
        "default": _search(query, _DS_DEFAULT),
    }


root_agent = Agent(
    model=_AGENT_MODEL,
    name="document_expert_agent",
    description="Vertex AI Search를 활용해 PDF 문서를 분석하는 에이전트",
    instruction=f"""당신은 전문 문서 분석 에이전트입니다.
                    기본 문서는 {_DEFAULT_DOCUMENT_GCS_URI}입니다.

                    사용자의 질문에 답하기 위해 아래 검색 도구를 적절히 선택하여 사용하십시오:
                    - search_layout: 표, 도표, 레이아웃 구조가 중요한 질문
                    - search_ocr: 정확한 원문 텍스트나 이미지 내 텍스트 검색
                    - search_default: 일반적인 문서 내용 검색
                    - compare_search: 세 파서의 결과를 비교하고 싶을 때

                    [중요] 검색 결과를 바탕으로 질문에 직접 답변하십시오.
                    "X 페이지를 참고하세요", "섹션 X에서 다루고 있습니다"처럼 위치만 안내하지 말고,
                    검색된 내용을 인용하여 실제 답변을 서술하십시오.""",
    tools=[search_layout, search_ocr, search_default, compare_search],
)
