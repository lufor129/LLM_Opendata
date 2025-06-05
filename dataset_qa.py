import os
import json
import pandas as pd
import requests
from typing import Annotated, Sequence, TypedDict, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv
import urllib3

# 加載環境變量
load_dotenv()

# 禁用 SSL 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# 定義狀態類型
class AgentState(TypedDict):
    dataset_info: Annotated[Dict[str, Any], "資料集信息"]
    csv_data: Annotated[pd.DataFrame, "CSV數據"]
    csv_summary: Annotated[str, "CSV摘要"]
    response: Annotated[str, "AI響應"]
    query: Annotated[str, "用戶查詢"]

# 初始化 LLM
llm = ChatOpenAI(model="o4-mini")

def download_csv(url: str, save_path: str) -> bool:
    """下載CSV文件
    
    Args:
        url: CSV文件URL
        save_path: 保存路徑
        
    Returns:
        bool: 是否下載成功
    """
    try:
        # 禁用 SSL 驗證
        response = requests.get(url, verify=False)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"下載CSV文件時出錯: {str(e)}")
        return False

def parse_column_info(column_info: str) -> Dict[str, str]:
    """解析欄位說明
    
    Args:
        column_info: 欄位說明字符串
        
    Returns:
        Dict[str, str]: 欄位名稱到說明的映射
    """
    result = {}
    for item in column_info.split(';'):
        if '(' in item:
            col_name, col_desc = item.split('(')
            result[col_name.strip()] = col_desc.strip(')')
    return result

def load_or_download_csv(dataset_info: Dict[str, Any]) -> pd.DataFrame:
    """加載或下載CSV文件
    
    Args:
        dataset_info: 資料集信息
        
    Returns:
        pd.DataFrame: CSV數據
    """
    # 創建CSV_DB目錄
    os.makedirs("CSV_DB", exist_ok=True)
    
    # 構建CSV文件路徑
    dataset_name = dataset_info["資料集名稱"]
    csv_filename = f"{dataset_name}.csv"
    csv_path = os.path.join("CSV_DB", csv_filename)
    
    # 如果文件不存在，則下載
    if not os.path.exists(csv_path):
        # 獲取所有URL並過濾出CSV格式的URL
        urls = dataset_info["資料下載網址"].split(';')

        # 使用 format=CSV 作為判斷條件
        csv_urls = [url for url in urls if 'format=csv' in url.lower() or "csv" in url.lower()]

        # if not csv_urls:
        #     raise Exception("找不到CSV格式的下載網址")
            
        if len(csv_urls) > 0:
            csv_url = csv_urls[0]  # 使用第一個CSV URL
        else:
            csv_url = urls[0]
        if not download_csv(csv_url, csv_path):
            raise Exception("無法下載CSV文件")
    
    # 讀取CSV文件
    df = pd.read_csv(csv_path)
    
    # 解析欄位說明
    column_info = parse_column_info(dataset_info["主要欄位說明"])
    
    # 重命名欄位
    df = df.rename(columns=column_info)
    
    return df

def summarize_csv(state: AgentState) -> AgentState:
    """生成CSV數據摘要
    
    Args:
        state: 當前狀態
        
    Returns:
        AgentState: 更新後的狀態
    """
    df = state["csv_data"]
    dataset_info = state["dataset_info"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個數據分析助手。請根據以下信息回答用戶的問題\n"
                  "資料集名稱：{dataset_name}\n"
                  "資料集描述：{dataset_desc}\n"
                  "數據內容：\n{data_preview}\n\n"
                  "請直接描述你根據用戶問題觀察到的數據特點\n"
                  "請用自然語言描述，不要生成代碼。"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({
        "dataset_name": dataset_info["資料集名稱"],
        "dataset_desc": dataset_info["資料集描述"],
        "data_preview": df.head().to_string(),
        "query": state["query"]
    })
    
    # 添加資料集連結
    dataset_id = dataset_info["資料集識別碼"]
    dataset_url = f"https://data.gov.tw/dataset/{dataset_id}"
    summary_with_link = f"{summary}\n\n您可以在以下網址查看完整資料集：\n{dataset_url}"
    
    return {"response": summary_with_link}

def generate_response(state: AgentState) -> AgentState:
    """生成回答
    
    Args:
        state: 當前狀態
        
    Returns:
        AgentState: 更新後的狀態
    """
    query = state["query"]
    df = state["csv_data"]
    dataset_info = state["dataset_info"]
    summary = state["csv_summary"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個數據分析助手。請根據以下信息回答用戶的問題：\n"
                  "資料集名稱：{dataset_name}\n"
                  "資料集描述：{dataset_desc}\n"
                  "數據摘要：{summary}\n"
                  "數據預覽：\n{data_preview}\n\n"
                  "請根據用戶的問題，結合數據內容給出準確的回答。"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "dataset_name": dataset_info["資料集名稱"],
        "dataset_desc": dataset_info["資料集描述"],
        "summary": summary,
        "data_preview": df.head().to_string(),
        "query": query
    })
    
    return {"response": response}

# 創建工作流
workflow = StateGraph(AgentState)

# 添加節點
workflow.add_node("summarize", summarize_csv)
workflow.add_node("generate", generate_response)

# 設置邊
workflow.add_edge(START, "summarize")
workflow.add_edge("summarize", END)
# workflow.add_edge("generate", END)

# 編譯工作流
app = workflow.compile()

def process_query(dataset_info: Dict[str, Any], query: str) -> str:
    """處理用戶查詢
    
    Args:
        dataset_info: 資料集信息
        query: 用戶查詢
        
    Returns:
        str: AI響應
    """
    try:
        # 加載或下載CSV
        df = load_or_download_csv(dataset_info)
        
        # 初始化狀態
        state = {
            "dataset_info": dataset_info,
            "csv_data": df,
            "csv_summary": "",
            "response": "",
            "query": query
        }
        
        # 運行工作流
        result = app.invoke(state)
        return result["response"]
    except Exception as e:
        return f"處理數據時發生錯誤：{str(e)}"

if __name__ == "__main__":
    # 測試數據
    dataset_info = {
        "資料集識別碼": "7276",
        "主要欄位說明": "no(序);area_name(區域);city_name(區域2);tour(遊程名稱);LocalCallService(市話)",
        "服務分類": "休閒旅遊",
        "資料下載網址": "https://cloud.hakka.gov.tw/Pub/Opendata/DTST20230500092.csv;https://cloud.hakka.gov.tw/Pub/Opendata/DTST20230500092.json;https://cloud.hakka.gov.tw/Pub/Opendata/DTST20230500092.xml",
        "資料集名稱": "2022桐花季經典45條桐花小旅行遊程",
        "資料集描述": "本會2022年補助地方政府辦理桐花祭活動核定名單。"
    }
    
    # 測試查詢
    test_queries = [
        "這個資料集包含哪些區域的遊程？",
        "請列出所有遊程名稱。"
    ]
    
    for query in test_queries:
        print(f"\n查詢: {query}")
        response = process_query(dataset_info, query)
        print(f"響應: {response}") 