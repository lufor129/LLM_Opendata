import streamlit as st
from rag_guardrail import RAGGuardrail
from dataset_qa import process_query as dataset_process_query
import pandas as pd
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re

# 載入環境變數
load_dotenv()

# 設置頁面配置
st.set_page_config(
    page_title="RAG Guardrail 開放資料對話系統",
    page_icon="🤖",
    layout="wide"
)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_suggestions" not in st.session_state:
    st.session_state.current_suggestions = {}
if "original_question" not in st.session_state:
    st.session_state.original_question = ""

def search_web(query: str) -> str:
    """使用 Google 搜索問題
    
    Args:
        query: 搜索查詢
        
    Returns:
        str: 搜索結果摘要
    """
    try:
        # 構建搜索 URL
        search_url = f"https://www.google.com/search?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        # 發送請求
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取搜索結果
        search_results = []
        for result in soup.find_all('div', class_='g')[:3]:  # 只取前3個結果
            title = result.find('h3')
            snippet = result.find('div', class_='VwiC3b')
            if title and snippet:
                search_results.append(f"標題：{title.text}\n摘要：{snippet.text}\n")
        
        if search_results:
            return "根據網絡搜索，我找到以下相關信息：\n\n" + "\n".join(search_results)
        else:
            return "抱歉，我無法找到相關的網絡信息。"
            
    except Exception as e:
        return f"搜索時發生錯誤：{str(e)}"

def display_chat_history():
    """顯示對話歷史"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

def display_suggestions(suggestions):
    """顯示建議按鈕"""
    if suggestions:
        st.write("您可以點擊以下建議繼續對話：")
        cols = st.columns(len(suggestions) + 1)  # 增加一列用於"堅持詢問原問題"按鈕
        
        # 顯示建議按鈕
        for idx, (key, suggestion) in enumerate(suggestions.items()):
            with cols[idx]:
                if st.button(suggestion, key=f"suggestion_{key}"):
                    return suggestion
        
        # 添加"堅持詢問原問題"按鈕
        with cols[-1]:
            if st.button("堅持詢問原問題", key="btn_original_question"):
                return "original_question"
    return None

def process_ai_response(prompt, rag_guardrail):
    """處理 AI 響應"""
    # 添加用戶消息到歷史
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 顯示用戶消息
    with st.chat_message("user"):
        st.write(prompt)
    
    # 獲取 AI 響應
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            if prompt == "original_question":
                # 如果用戶選擇堅持詢問原問題，進行網絡搜索
                AI_feedback = search_web(st.session_state.original_question)
                st.session_state.current_suggestions = {}
            else:
                gl_state = rag_guardrail.process_query(prompt)
                if not gl_state["is_allowed"]:
                    AI_feedback = "沒有找到完全匹配的答案，以下是一些相關的問題建議："
                    st.warning(AI_feedback)
                    st.session_state.current_suggestions = gl_state["suggestions"]
                    st.session_state.original_question = prompt  # 保存原始問題
                else:
                    # 如果找到匹配的資料集，使用 dataset_qa 處理查詢
                    if gl_state["target_dataset"] != "其他":
                        try:
                            dataset_info = {
                                "資料集名稱": gl_state["target_dataset"].metadata.get("資料集名稱", ""),
                                "資料集描述": gl_state["target_dataset"].metadata.get("資料集描述", ""),
                                "資料下載網址": gl_state["target_dataset"].metadata.get("資料下載網址", ""),
                                "主要欄位說明": gl_state["target_dataset"].metadata.get("主要欄位說明", ""),
                                "資料集識別碼": gl_state["target_dataset"].metadata.get("資料集識別碼", "")
                            }
                            print(gl_state["target_dataset"] )
                            AI_feedback = dataset_process_query(dataset_info, prompt)
                        except Exception as e:
                            AI_feedback = f"處理數據時發生錯誤：{str(e)}"
                    else:
                        AI_feedback = gl_state["intention"]
                    st.session_state.current_suggestions = {}
                st.write(AI_feedback)
    
    # 添加 AI 響應到歷史
    st.session_state.messages.append({"role": "assistant", "content": AI_feedback})

def main():
    rag_guardrail = RAGGuardrail()
    st.title("🤖 對話系統")

    # 主界面 - 對話區域
    st.header("💬 開放資料對話機器人")
    
    # 顯示對話歷史
    display_chat_history()
    
    # 顯示建議按鈕
    if st.session_state.current_suggestions:
        selected_suggestion = display_suggestions(st.session_state.current_suggestions)
        if selected_suggestion:
            process_ai_response(selected_suggestion, rag_guardrail)
            st.rerun()
    
    # 用戶輸入
    if prompt := st.chat_input("請輸入您的問題"):
        process_ai_response(prompt, rag_guardrail)
        st.rerun()
    
    # 添加清除對話按鈕
    if st.button("清除對話歷史"):
        st.session_state.messages = []
        st.session_state.current_suggestions = {}
        st.session_state.original_question = ""
        st.rerun()

if __name__ == "__main__":
    main() 