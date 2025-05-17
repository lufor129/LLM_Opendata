from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START
import os
from dotenv import load_dotenv
import re
from difflib import SequenceMatcher

# 加載環境變量
load_dotenv()

# 定義狀態類型
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "對話歷史"]
    dataset_name: Annotated[str, "使用者想搜尋的目標資料集"]
    intention: Annotated[str, "使用者對資料集的意圖"]
    context: Annotated[str, "檢索到的上下文"]
    summary: Annotated[str, "CSV內容總結"]
    response: Annotated[str, "AI 響應"]
    is_allowed: Annotated[bool, "是否允許回答"]
    retry_count: Annotated[int, "重試次數"]
    should_terminate: Annotated[bool, "是否應該終止對話"]

# 初始化 LLM
llm = ChatOpenAI(model="o4-mini")

persist_directory = "chroma_db"

# 初始化向量存储
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=persist_directory 
)

# 定義 RAG 檢索函數
def retrieve_context(state: AgentState) -> AgentState:
    # 获取最后一条用户消息
    last_message = state["messages"][-1].content
    
    # 从向量存储中检索相关文档
    docs = vectorstore.similarity_search(last_message, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    
    return {"context": context}

# 定義問題分析函數
def analyze_question(query: str) -> tuple[str, str]:
    """分析用戶問題，提取目標資料集和意圖
    
    Args:
        query: 用戶問題
        
    Returns:
        tuple[str, str]: (目標資料集, 意圖)
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個問題分析器。請將用戶的問題分解為目標資料集和意圖。\n"
                  "目標資料集：用戶想要查詢的具體資料集名稱\n"
                  "意圖：用戶想要對該資料集執行的操作\n\n"
                  "請用JSON格式返回，格式如下：\n"
                  "{{\n"
                  '  "dataset": "目標資料集",\n'
                  '  "intention": "意圖"\n'
                  "}}"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"query": query})
    
    try:
        import json
        result = json.loads(response)
        return result["dataset"], result["intention"]
    except:
        return query, "查詢"

def calculate_iou(str1: str, str2: str) -> float:
    """計算兩個字符串的 IOU 相似度
    
    Args:
        str1: 第一個字符串
        str2: 第二個字符串
        
    Returns:
        float: IOU 相似度 (0-1)
    """
    # 將字符串轉換為字符集合
    set1 = set(str1)
    set2 = set(str2)
    
    # 計算交集和並集
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    # 計算 IOU
    return intersection / union if union > 0 else 0

def fuzzy_match(pattern: str, text: str, threshold: float = 0.5) -> bool:
    """使用正則表達式和 IOU 進行模糊匹配
    
    Args:
        pattern: 匹配模式
        text: 待匹配文本
        threshold: IOU 閾值，默認為 0.5
        
    Returns:
        bool: 是否匹配成功
    """
    # 將模式轉換為正則表達式
    pattern = re.escape(pattern)
    pattern = pattern.replace('\\ ', '.*?')  # 允許空格之間有任意字符
    
    # 嘗試正則匹配
    if re.search(pattern, text, re.IGNORECASE):
        return True
    
    # 如果正則匹配失敗，計算 IOU
    return calculate_iou(pattern, text) >= threshold

# 定義資料集相似函數
def match_dataset(target_dataset: str) -> tuple[bool, str, list[str], list[dict]]:
    """匹配目標資料集與RAG數據庫中的資料集
    
    Args:
        target_dataset: 目標資料集名稱
        
    Returns:
        tuple[bool, str, list[str], list[dict]]: (是否匹配, 匹配的資料集名稱, 相關資料集列表, 相關元數據列表)
    """
    # 使用 RAG 檢索相關文檔
    docs = vectorstore.similarity_search(target_dataset, k=3)
    
    # 獲取相關資料集名稱
    related_datasets, related_metas = [], []
    for doc in docs:
        dataset_name = doc.metadata.get("資料集名稱", "")
        if dataset_name and dataset_name not in related_datasets:
            related_datasets.append(dataset_name)
            related_metas.append(doc.metadata)
    
    # 使用模糊匹配判斷是否相似
    is_match = False
    matched_dataset = ""
    
    for dataset in related_datasets:
        if fuzzy_match(target_dataset, dataset):
            is_match = True
            matched_dataset = dataset
            break
    print(related_metas)
    print("matched_dataset", matched_dataset)
    return is_match, matched_dataset, related_datasets, related_metas

# 定義 Guardrail 檢查函數
def check_guardrail(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    
    # 檢查是否包含終止關鍵詞
    if any(keyword in query.lower() for keyword in ["終止", "停止", "結束", "quit", "stop", "exit"]):
        return {
            "is_allowed": False,
            "should_terminate": True,
            "response": "好的，已終止服務。感謝您的使用！"
        }
    
    # 分析問題
    target_dataset, intention = analyze_question(query)
    
    # 匹配資料集
    is_match, matched_dataset, context, related_metas = match_dataset(target_dataset)
    print("is match", is_match)
    print("matched_dataset", matched_dataset)
    
    if not is_match:
        retry_count = state.get("retry_count", 0) + 1
        if retry_count >= 3:
            return {
                "is_allowed": False,
                "should_terminate": True,
                "retry_count": retry_count,
                "response": "抱歉，經過多次嘗試後仍無法找到相關資料集。服務已終止。"
            }
        
        # 構建反饋信息
        feedback = "抱歉，我找不到完全匹配的資料集。\n\n"
        for metadata in related_metas:
            feedback += f"我找到的相關資料集：\n{metadata.get('資料集名稱', '')} \n\n"
        feedback += "請確認您要查詢的資料集名稱，或輸入'終止'來結束對話。"
        
        return {
            "is_allowed": False,
            "should_terminate": False,
            "retry_count": retry_count,
            "response": feedback,
            "dataset_name": target_dataset,
            "intention": intention
        }
    
    # 如果完全匹配，更新狀態
    return {
        "is_allowed": True,
        "should_terminate": False,
        "retry_count": 0,
        "dataset_name": matched_dataset,
        "intention": intention,
        "context": context
    }

# 新增 summary agent 節點，對 context（CSV文本）做總結
def summary_agent(state: AgentState) -> AgentState:
    if not state.get("is_allowed", False):
        return {"summary": ""}
    context = state.get("context", "")
    if not context.strip():
        return {"summary": "未檢索到相關CSV內容。"}
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個CSV摘要助手。請對以下CSV內容做簡要總結，提取關鍵信息：\n{csv_content}"),
        ("human", "請用簡潔的中文總結上面的CSV內容。")
    ])
    chain = prompt | llm | StrOutputParser()
    summary = chain.invoke({"csv_content": context})
    return {"summary": summary}

# 修改生成響應的函數
def generate_response(state: AgentState) -> AgentState:
    if state.get("should_terminate", False):
        return state
    
    if not state.get("is_allowed", False):
        return state
    
    dataset_name = state.get("dataset_name", "")
    intention = state.get("intention", "")
    context = state.get("context", "")
    
    # 根據意圖生成響應
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個數據分析助手。請根據用戶的意圖分析數據。\n"
                  "資料集：{dataset}\n"
                  "意圖：{intention}\n"
                  "數據內容：\n{context}"),
        ("human", "請根據以上信息回答用戶的問題。")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "dataset": dataset_name,
        "intention": intention,
        "context": context
    })
    
    return {"response": response}

# 修改工作流圖
workflow = StateGraph(AgentState)

# 添加節點
workflow.add_node("retrieve", retrieve_context)
workflow.add_node("check_guardrail", check_guardrail)
workflow.add_node("summarize_agent", summary_agent)
workflow.add_node("generate", generate_response)

# 設置邊
workflow.add_edge(START, "retrieve")  # 使用 START 常量
workflow.add_edge("retrieve", "check_guardrail")
workflow.add_edge("check_guardrail", "summarize_agent")
workflow.add_edge("summarize_agent", "generate")
workflow.add_edge("generate", END)

# 編譯工作流
app = workflow.compile()

# 修改示例使用函數
def process_query(query: str):
    # 初始化狀態
    state = {
        "messages": [HumanMessage(content=query)],
        "dataset_name": "",
        "intention": "",
        "context": "",
        "summary": "",
        "response": "",
        "is_allowed": True,
        "retry_count": 0,
        "should_terminate": False
    }
    
    # 運行工作流
    result = app.invoke(state)
    return result["response"]

# 添加文件到向量存储的函數
def add_documents(documents: list[str]):
    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # 創建文件
    docs = [Document(page_content=text) for text in documents]
    splits = text_splitter.split_documents(docs)
    
    # 添加到向量存储
    vectorstore.add_documents(splits)

if __name__ == "__main__":
    # 測試查詢
    test_queries = [
        # "客家桐花在幾月的時候開？",  # first
        "2022桐花季經典45條桐花小旅行遊程有哪些？"
    ]
    
    for query in test_queries:
        print(f"\n查詢: {query}")
        response = process_query(query)
        print(f"響應: {response}") 