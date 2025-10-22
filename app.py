# app.py
# Python 3.11 推奨
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# --- .env から環境変数を読み込み（ローカル開発用） ---
load_dotenv()  # ← これで .env ファイルの OPENAI_API_KEY を読み込む

st.set_page_config(page_title="LangChain × Streamlit Demo", page_icon="💬")

# ---------- 説明 ----------
st.title("💬 LangChain × Streamlit：専門家モードQA")
st.markdown(
    """
**概要**  
入力フォームに質問や相談を書き、下のラジオボタンで「専門家の種類」を選んで送信すると、  
LangChain 経由で LLM にプロンプトを渡し、選択した専門家として回答が返ってきます。  

**使い方**  
1. 専門家の種類を選ぶ（A か B）  
2. 入力欄に質問やテキストを記入  
3. **送信** ボタンを押す  
"""
)

# ---------- LLM初期化 ----------
@st.cache_resource
def load_llm() -> ChatOpenAI:
    """
    ChatOpenAI のインスタンスを初期化して返す。
    環境変数または secrets.toml から OPENAI_API_KEY を読み取る。
    """
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or st.secrets.get("OPENAI_API_KEY", "")
    )
    if not api_key:
        st.warning(
            "⚠️ OPENAI_API_KEY が設定されていません。\n"
            "`.env` ファイルまたは `.streamlit/secrets.toml` にキーを設定してください。"
        )

    return ChatOpenAI(model_name="gpt-4o-mini", temperature=0, api_key=api_key)

# ---------- 回答生成関数 ----------
def generate_response(input_text: str, persona_choice: str) -> str:
    """
    入力テキストとラジオボタン選択値を受け取り、
    選択された専門家として LLM の回答を返す。
    """
    system_prompts = {
        "A. ソフトウェアアーキテクト":
            "あなたは経験豊富なソフトウェアアーキテクトです。"
            "要件整理→設計方針→技術選定→リスク→実装手順の順で簡潔に答えてください。",
        "B. マーケティング戦略家":
            "あなたはデータドリブンなマーケティング戦略家です。"
            "ターゲット→施策→KPI→検証方法を明確に提案してください。",
    }

    system_msg = system_prompts.get(persona_choice, "あなたは丁寧なアシスタントです。")

    llm = load_llm()
    messages = [
        SystemMessage(content=system_msg),
        HumanMessage(content=input_text),
    ]
    result = llm.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)

# ---------- 入力フォーム ----------
with st.form(key="qa_form"):
    persona = st.radio(
        "専門家の種類を選択してください：",
        ["A. ソフトウェアアーキテクト", "B. マーケティング戦略家"]
    )
    user_input = st.text_area(
        "質問・相談を入力してください：",
        height=160,
        placeholder="例：小規模チームでのBtoB SaaS を短期間で構築するには？"
    )
    submitted = st.form_submit_button("送信")

# ---------- 結果表示 ----------
if submitted:
    if not user_input.strip():
        st.error("⚠️ テキストを入力してください。")
    else:
        with st.spinner("LLM に問い合わせ中..."):
            try:
                answer = generate_response(user_input, persona)
                st.markdown("### 🧠 回答")
                st.write(answer)
            except Exception as e:
                st.error(f"エラーが発生しました: {e}")

# ---------- フッター ----------
st.divider()
st.caption(
    "・Python 3.11 で動作確認済み。\n"
    "・モデル: `gpt-4o-mini`"
)
