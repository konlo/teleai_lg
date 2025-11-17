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
