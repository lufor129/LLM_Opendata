import streamlit as st
from rag_guardrail import process_query
from excel_to_rag import ExcelToRAG
import pandas as pd
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 設置頁面配置
st.set_page_config(
    page_title="RAG Guardrail 對話系統",
    page_icon="🤖",
    layout="wide"
)

# 初始化 session state
if "messages" not in st.session_state:
    st.session_state.messages = []

def main():
    st.title("🤖 RAG Guardrail 對話系統")
    
    # 側邊欄 - 文件上傳
    with st.sidebar:
        st.header("📁 數據導入")
        uploaded_file = st.file_uploader("上傳 Excel 文件", type=["xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                # 保存上傳的文件
                with open("temp.xlsx", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # 處理 Excel 文件
                excel_to_rag = ExcelToRAG()
                excel_to_rag.process_excel("temp.xlsx")
                
                st.success("Excel 文件處理成功！")
                
                # 顯示數據預覽
                df = pd.read_excel("temp.xlsx")
                st.subheader("數據預覽")
                st.dataframe(df.head())
                
                # 清理臨時文件
                os.remove("temp.xlsx")
                
            except Exception as e:
                st.error(f"處理文件時出錯: {str(e)}")
    
    # 主界面 - 對話區域
    st.subheader("💬 對話")
    
    # 顯示歷史消息
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # 用戶輸入
    if prompt := st.chat_input("請輸入您的問題"):
        # 添加用戶消息到歷史
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 顯示用戶消息
        with st.chat_message("user"):
            st.write(prompt)
        
        # 獲取 AI 響應
        with st.chat_message("assistant"):
            with st.spinner("思考中..."):
                response = process_query(prompt)
                st.write(response)
        
        # 添加 AI 響應到歷史
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 添加清除對話按鈕
    if st.button("清除對話歷史"):
        st.session_state.messages = []
        st.experimental_rerun()

if __name__ == "__main__":
    main() 