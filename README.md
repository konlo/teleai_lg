2025.11.26일 수정 

핵심은 시스템이 사용자 입력의 의도를 더 잘 파악하거나, 사용자가 명시적으로 의도를 선택하도록 유도하는 것입니다.
1. 사용자 프롬프트 가이드라인 제시 및 예시 제공
가장 간단한 방법은 사용자 입력창 옆이나 도움말 섹션에 명확한 가이드라인을 제공하는 것입니다.
전체 데이터 시각화 (DataFrame 직접 로드) 유도:
사용자에게 특정 열, 필터 조건, 정렬 기준 등 세부적인 분석 요구사항을 입력하도록 유도합니다.
예시 프롬프트: "지난주 특정 제품의 시간별 판매량을 라인 차트로 보여줘", "지역별 고객 분포를 지도 시각화로 나타내줘"
통계 데이터 시각화 (SQL 추출) 유도:
사용자에게 요약, 추세, 순위, 집계 등을 요청하는 키워드를 사용하도록 안내합니다. 이는 대규모 데이터셋에 효율적입니다.
예시 프롬프트: "전체 기간 동안 가장 많이 팔린 제품 TOP 10을 막대 그래프로 보여줘", "월별 총 매출 추세를 분석해줘", "평균 구매 금액을 파이 차트로 나타내줘"
2. 시스템의 입력 의도 분석 로직 강화
사용자 프롬프트에 포함된 특정 키워드를 기반으로 시스템 내부에서 자동으로 시나리오를 분류하도록 로직을 개선할 수 있습니다.
키워드 매핑:
SQL/통계 처리 유도 키워드: '평균', '합계', '최대', '최소', 'TOP N', '총', '추세', '분포', '비율', '집계' 등이 포함되면 통계적 접근을 기본값으로 설정합니다.
DataFrame 처리 유도 키워드: '모든', '세부', '시간별', '분 단위', '필터링', '원시(raw) 데이터' 등이 포함되면 DataFrame 로드를 시도합니다.
데이터 크기 자동 확인 (예외 처리):
사용자가 세부 시각화를 요청하더라도, 대상 테이블의 데이터 크기가 너무 크다면(예: 1억 행 이상) 사용자에게 알림을 제공하거나, 자동으로 통계적 접근 방식으로 전환하도록 설계할 수 있습니다.
시스템 메시지 예시: "요청하신 데이터 양이 너무 많아 전체 로딩이 어렵습니다. 대신, 통계적 요약 데이터를 기반으로 시각화하겠습니다."
3. 사용자 인터페이스(UI)에서 명시적 선택 옵션 제공
가장 확실한 방법은 사용자에게 선택권을 주는 것입니다. 텍스트 프롬프트 입력 외에 별도의 UI 요소를 추가합니다.
시각화 모드 선택 스위치/드롭다운 메뉴:
프롬프트 입력창 상단이나 하단에 "시각화 모드 선택" 옵션을 제공합니다.
[세부 데이터 분석 (DataFrame)] 또는 [대규모 데이터 요약 (SQL 통계)]
선택적 확인 질문:
사용자가 모호한 프롬프트를 입력했을 때, 챗봇이 되물어볼 수 있습니다.
챗봇: "이 요청은 대규모 데이터 집계가 필요할 수 있습니다. 전체 세부 데이터를 불러와서 분석하시겠습니까, 아니면 SQL로 추출한 요약 통계 데이터를 사용하시겠습니까?"
요약 및 권장사항
세 가지 방법 중 **"1. 가이드라인 제시"**와 **"2. 시스템의 의도 분석 로직 강화"**를 병행하는 것이 시작점으로 가장 좋습니다. 사용자의 편의성을 해치지 않으면서도 시스템이 더 스마트하게 작동하도록 만들 수 있습니다.
데이터 크기에 따라 자동으로 처리 방식을 전환하는 로직(2번의 예외 처리)을 추가하면 사용자가 구분을 어려워하는 문제를 효과적으로 완화할 수 있습니다.



# teleai_lg

## LangGraph Example

`examples/langgraph_example.py` runs a compact LangGraph workflow that summarizes any topic via your chosen LLM provider.

### How to run
- Install deps: `pip install langgraph streamlit openai google-generativeai python-dotenv`
- Set your Google Gemini key (default provider): `export GOOGLE_API_KEY=...`
- (Optional) use OpenAI instead: `export LLM_PROVIDER=openai && export OPENAI_API_KEY=sk-...`
- (Optional) use Azure OpenAI: set `LLM_PROVIDER=azure` and configure `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` (plus optional `AZURE_OPENAI_API_VERSION`).
- (Optional) Choose a topic: `export TOPIC="multimodal AI"`
- Run the script: `python examples/langgraph_example.py`
- Both the script and Streamlit app load environment variables from `.env` automatically via `python-dotenv`.

## Streamlit Chatbot

`streamlit_app.py` launches a LangGraph-powered chat UI; it reads `LLM_PROVIDER` from your environment (default `google`) to decide whether to call Gemini, OpenAI, or Azure OpenAI.

### Launch
- Install deps (see above).
- Export `LLM_PROVIDER` (defaults to `google`) and the matching credentials: `GOOGLE_API_KEY` (plus optional `GOOGLE_MODEL`), `OPENAI_API_KEY` (plus optional `OPENAI_MODEL`), or Azure values `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT` (plus optional `AZURE_OPENAI_API_VERSION`).
- Start Streamlit: `streamlit run streamlit_app.py`
