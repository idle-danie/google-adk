import json
import os

from dotenv import load_dotenv
from google.adk.agents import Agent
from google.cloud import documentai_v1 as documentai, storage
from google.api_core.client_options import ClientOptions

load_dotenv()

_PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
_DOCAI_LOCATION = os.environ["DOCAI_LOCATION"]
_PROCESSOR_ID = os.environ["DOCAI_PROCESSOR_ID"]
_OUTPUT_GCS_URI = os.environ["DOCAI_OUTPUT_GCS_URI"]
_AGENT_MODEL = os.environ["AGENT_MODEL"]
_DEFAULT_DOCUMENT_GCS_URI = os.environ["DEFAULT_DOCUMENT_GCS_URI"]


def process_document_from_gcs(gcs_uri: str) -> dict:
    """
    GCS에 저장된 PDF 문서를 Document AI 배치 처리로 분석하여 텍스트를 추출합니다.

    Args:
        gcs_uri (str): 분석할 GCS 경로 (예: 'gs://your-bucket/file.pdf')

    Returns:
        dict: status와 추출된 텍스트 또는 오류 메시지
    """
    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(api_endpoint=f"{_DOCAI_LOCATION}-documentai.googleapis.com")
    )
    resource_name = client.processor_path(_PROJECT_ID, _DOCAI_LOCATION, _PROCESSOR_ID)

    request = documentai.BatchProcessRequest(
        name=resource_name,
        input_documents=documentai.BatchDocumentsInputConfig(
            gcs_documents=documentai.GcsDocuments(
                documents=[documentai.GcsDocument(gcs_uri=gcs_uri, mime_type="application/pdf")]
            )
        ),
        document_output_config=documentai.DocumentOutputConfig(
            gcs_output_config=documentai.DocumentOutputConfig.GcsOutputConfig(
                gcs_uri=_OUTPUT_GCS_URI
            )
        ),
    )

    operation = client.batch_process_documents(request=request)
    operation.result(timeout=600)

    # GCS 출력 결과 읽기
    bucket_name = _OUTPUT_GCS_URI.split("/")[2]
    prefix = "/".join(_OUTPUT_GCS_URI.split("/")[3:])

    storage_client = storage.Client()
    blobs = list(storage_client.bucket(bucket_name).list_blobs(prefix=prefix))

    texts = []
    for blob in blobs:
        if blob.name.endswith(".json"):
            content = json.loads(blob.download_as_text())
            texts.append(content.get("text", ""))

    return {"status": "success", "text": "\n".join(texts)}


root_agent = Agent(
    model=_AGENT_MODEL,
    name="document_expert_agent",
    description="GCS에 저장된 PDF 문서를 Document AI로 분석하는 에이전트",
    instruction=f"""
                    당신은 전문 문서 분석 에이전트입니다.
                    사용자가 문서에 대해 질문하면 process_document_from_gcs 도구를 사용해 내용을 추출하고 답변하십시오.
                    기본 문서는 {_DEFAULT_DOCUMENT_GCS_URI} 입니다.
                    사용자가 다른 URI를 언급하지 않으면 이 문서를 사용하십시오.
                """,
    tools=[process_document_from_gcs],
)
