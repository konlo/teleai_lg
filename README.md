-----------
2025.11.30
-----------
Prompt 시작

당신은 Streamlit + LangGraph 기반 chatbot을 개선하는 Senior Python Engineer입니다.
아래 요구사항을 만족하도록 **“기존에 이미 존재하는 Streamlit chatbot 코드 안에 기능을 추가”**하세요.
새로운 앱을 만들지 말고, 기존 chatbot 코드 구조를 유지하고 필요한 부분만 확장/삽입하세요.

🎯 리팩토링 목적

지금 내가 사용 중인 chatbot 앱에
%debug on/off 토글 + astream_events 기반 debug 모니터링 기능을 추가하는 것이다.

기존 chatbot 코드의 흐름, UI, 노드 구성, 그래프 실행 방식 등은 절대로 변경하지 마라.

단지 기존 코드 안에 필요한 함수, if-logic, sidebar 출력만 자연스럽게 삽입해라.

🧩 기능 요구사항
1) %debug on/off 명령 처리

사용자 입력이 %debug on → st.session_state.debug_mode = True

사용자 입력이 %debug off → st.session_state.debug_mode = False

이 두 명령은 “일반 질문”으로 처리하지 않고 내부 명령으로만 처리

명령을 처리한 경우 chatbot 실행 로직은 호출하지 않는다.

2) 기존 chatbot 실행 로직에 조건부 이벤트 스트리밍 삽입

기존 chatbot은 LangGraph app.invoke() 또는 app.astream() 등을 사용해 실행 중일 것이다.

debug_mode = False → 기존 로직 그대로 유지, 아무 것도 바꾸지 마라.

debug_mode = True → 기존 실행 대신 await app.astream_events() 를 삽입하여 실행.

예:
if debug_mode:
    async for event in app.astream_events({"input": user_query}, version="v1"):
        handle_event(event)
else:
    result = app.invoke({"input": user_query})


⚠ 기존의 실행 체인, state 업데이트, 메인 화면 메시지 출력 로직은 절대 삭제하지 말고 그대로 연동하라.

3) Sidebar에 Debug Events 출력 (Debug ON일 때만)

sidebar 영역 제목: "🔍 Debug Events"

출력해야 하는 이벤트:

node.started

node.completed

llm.streaming.chunk

tool.started

tool.completed

state.diff (중요)

예시 출력:

🟡 Node Started: node_extract_user
🟢 Node Completed: node_select_table
✏️ Token: "The"
🔧 Tool Started: run_sql_tool
🔨 Tool Completed: run_sql_tool
🧩 State Diff:
{
    “selected_table”: “telemetry_v1”
}


debug OFF일 때는 sidebar의 debug 영역 전체를 렌더링하지 않는다.

🧰 기술 요구사항
반드시 아래 helper 함수들을 “기존 코드 안에 삽입하라”

(새로운 앱 생성 금지)

parse_debug_command(text)

render_debug_sidebar()

async run_graph_with_events(app, query, debug_box)

기존 run_graph 로직은 건드리지 말고, debug_mode일 때만 override 하도록 조건부 실행

Async runner 유틸 추가 (기존 코드가 async/await 혼용 시 필요)
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if loop.is_running():
        return asyncio.ensure_future(coro)
    return asyncio.run(coro)

🧱 코드 삽입 위치 가이드

Codex는 아래 원칙을 반드시 지켜야 한다:

기존 chatbot의 main 함수, 로직 흐름, LangGraph app 정의, 노드, 상태, LLM 호출 로직은 절대 수정하거나 삭제하지 마라.

단지:

사용자의 입력 처리 부분

LangGraph 실행 호출 부분(app.invoke/app.run 등)

Streamlit sidebar 부분
에 조건부로 debug 기능을 추가하라.

📌 최종 산출물

Codex가 출력하는 코드는 새로운 앱을 만들지 말고,
“내가 이미 가지고 있는 chatbot 코드에 삽입해야 할 코드 조각 + 수정되는 부분”을 모든 위치와 함께 정확하게 제시해야 한다.

즉, “여기 아래에 붙이세요”, “이 부분을 이렇게 감싸세요”, “이 함수 바로 위에 넣으세요” 형태로 지시할 것.

기존 chatbot 코드를 유지하면서 필요한 기능을 자연스럽게 추가하는 형태의 완전한 Patch 코드를 생성하라.

Prompt 종료

----------------
2025.11.27일 수정
----------------
현재 그래프는 확실히 노드가 너무 많고 엣지도 복잡해서,

새로운 기능 추가할 때마다 노드+엣지 추가 → 유지보수 지옥
디버깅할 때 어디로 가는지 따라가기 힘듦
상태 전이 로직이 분산되어 있어서 전체 흐름 파악이 어려움
이건 전형적인 “작은 단계마다 노드 하나씩” 설계의 부작용입니다.
LangGraph는 StateGraph라서 같은 기능을 하는 여러 단계를 하나의 노드로 묶어도 전혀 문제없습니다.

현재 17개 노드 → 목표 8~10개 수준으로 압축 가능
아래는 실제 운영 중인 여러 프로덕션 LangGraph SQL/Visualization 에이전트들을 참고해서 정리한, 가장 단순하면서도 확장성 높은 구조입니다.
결론 & 추천 행동 계획
당장 리팩토링해도 충분히 가치 있습니다.

위 8개 노드 구조로 새 그래프 하나 만들어서 돌려보기 (기존 그래프는 그대로 두고 A/B처럼)
하나씩 기존 노드 로직을 새 노드 안으로 옮기면서 테스트
2주 안에 17→8 노드로 전환 가능 (실제 제가去年에 22개 노드 에이전트를 9개로 줄였을 때 걸린 시간)
이 정도만 정리해도 새 기능 추가 속도가 3~5배 빨라지고, 팀원 누구나 그래프 전체 흐름을 10분 안에 이해할 수 있게 됩니다.

필요하면 제가 실제 코드 레벨에서 8개 노드 버전으로 통합해드릴 수도 있어요!

--------
추가 정리
--------
LangGraph에서 node는

기능(Function)이 아니라 “AI에게 맡길 행동 단위(Behavior)”
그리고 **“분기(decision)가 발생하는 지점”**을 기준으로 설계해야 한다.

원하면 네 그래프 전체를
Behavior 기준으로 재설계한 6~7개 노드 버전으로 깔끔하게 리팩토링해줄게.


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
