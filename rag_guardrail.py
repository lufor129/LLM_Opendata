from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END, START
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

# 定義狀態類型
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "對話歷史"]
    dataset_name: Annotated[str, "使用者想搜尋的目標資料集"]
    target_dataset: Annotated[Document, "目標資料集"]
    intention: Annotated[str, "使用者對資料集的意圖"]
    quert_documents: Annotated[Sequence[Document], "檢索到的文件標題們"]
    suggestions: Annotated[dict, "可能的問題建議"]
    response: Annotated[str, "AI 響應"]
    is_allowed: Annotated[bool, "是否允許回答"]
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
    query = state["messages"][-1].content
    
    docs = vectorstore.similarity_search(query, k=3)
    return {"quert_documents": docs}

# 定義問題分析函數
def analyze_question(state: AgentState) -> AgentState:
    """分析用戶問題，提取目標資料集和意圖
    
    Args:
        state: 狀態對象，包含用戶問題和檢索到的上下文
        
    Returns:
        AgentState: 更新後的狀態對象，包含目標資料集和意圖
    """
  
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個問題分析器。請將用戶的問題分解為目標資料集和意圖。\n"
                  "目標資料集：用戶想要查詢的具體資料集名稱\n"
                  "意圖：用戶想要對該資料集執行的操作\n\n"
                  "目標資料集可以參考以下列表：\n{context_titles_str}\n\n"
                  "若列表找不到相關資料集，請用'其他'來表示\n\n"
                  "請用JSON格式返回，格式如下：\n"
                  "{{\n"
                  '  "dataset": "目標資料集",\n'
                  '  "intention": "意圖"\n'
                  "}}"),
        ("human", "{query}")
    ])
    query = state["messages"][-1].content
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({
        "query": query,
        "context_titles_str": "\n".join([doc.metadata.get("資料集名稱", "") for doc in state["quert_documents"]])
    })

    # 將 response 轉換為 JSON 格式
    try:
        import json
        result = json.loads(response)
    except:
        dataset_name = "其他"
        intention = "查詢"
        result = {"dataset": dataset_name}

    if result["dataset"] != "其他":
        match_index = [doc.metadata.get("資料集名稱", "") for doc in state["quert_documents"]].index(result["dataset"])
        target_dataset = state["quert_documents"][match_index]
        dataset_name = result["dataset"]
        intention = result["intention"]
    else:
        target_dataset = "其他"
        dataset_name = "其他"
        intention = "查詢"


    return {
        "dataset_name": dataset_name,
        "intention": intention,
        "target_dataset": target_dataset
    }

def generate_question_suggestions(state: AgentState) -> AgentState:
    """根據檢索到的文檔生成問題建議
    
    Args:
        state: 當前狀態
        
    Returns:
        AgentState: 更新後的狀態，包含問題建議
    """
    if state["target_dataset"] != "其他":
        return state
        
    # 獲取所有檢索到的文檔標題
    dataset_title_description = [
        "Title: {}, desceiption: {}\n\n".format(doc.metadata.get("資料集名稱", ""), doc.metadata.get("資料集描述", ""))
        for doc in state["quert_documents"]]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一個問題生成助手。請根據以下資料集標題，生成3個可能的用戶問題。\n"
                  "這些問題應該：\n"
                  "1. 生成的問題盡量包含資料集Title\n"
                  "2. 使用自然的口語表達\n"
                  "3. 針對資料集的具體內容\n"
                  "4. 符合一般用戶的查詢習慣\n\n"
                  "資料集標題與描述：\n{dataset_title_description}\n\n"
                  "請用JSON格式返回，格式如下：\n"
                  "{{\n"
                  '  "suggestions": {{\n'
                  '    "資料集1": "問題1",\n'
                  '    "資料集2": "問題2",\n'
                  '    "資料集3": "問題3"\n'
                  '  }}\n'
                  "}}"),
        ("human", "請生成問題建議")
    ])
    
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"dataset_title_description": "\n".join(dataset_title_description)})
    
    try:
        import json
        result = json.loads(response)
        suggestions = result["suggestions"]
        
        # 構建建議信息
        suggestion_text = "我找到以下相關資料集，您可以嘗試詢問：\n\n"
        for dataset, question in suggestions.items():
            suggestion_text += f"{dataset}. {question}\n"
        
        return {
            "is_allowed": False,
            "should_terminate": False,
            "response": suggestion_text,
            "suggestions": suggestions
        }
    except:
        return {
            "is_allowed": False,
            "should_terminate": True,
            "response": "抱歉，我無法生成問題建議。",
            "suggestions": {}
        }

# 修改 check_guardrail 函數
def check_guardrail(state: AgentState) -> AgentState:
    query = state["messages"][-1].content
    # 檢查是否包含終止關鍵詞
    if any(keyword in query.lower() for keyword in ["終止", "停止", "結束", "quit", "stop", "exit"]):
        return {
            "is_allowed": False,
            "should_terminate": True,
            "response": "好的，已終止服務。感謝您的使用！"
        }
    
    if state["dataset_name"] == "其他":
        return generate_question_suggestions(state)
    else:
        return generate_response(state)

# 修改生成響應的函數
def generate_response(state: AgentState) -> AgentState:
    if state.get("should_terminate", False):
        return state
    
    if not state.get("is_allowed", False):
        return state
    
    dataset_name = state.get("dataset_name", "")
    intention = state.get("intention", "")
    target_dataset = state.get("target_dataset", None)
    
    if target_dataset is None:
        return {
            "is_allowed": False,
            "should_terminate": True,
            "response": "抱歉，無法找到相關資料集。"
        }
    else:
        return {
            "is_allowed": True,
            "should_terminate": False,
            "response": target_dataset.page_content
        }
    
class RAGGuardrail:
    def __init__(self):
        self.workflow = StateGraph(AgentState)

        # 修改工作流圖
        workflow = StateGraph(AgentState)

        # 添加節點
        workflow.add_node("retrieve", retrieve_context)
        workflow.add_node("check_guardrail", check_guardrail)
        workflow.add_node("analyze_question", analyze_question)
        workflow.add_node("generate_question_suggestions", generate_question_suggestions)
        workflow.add_node("generate_response", generate_response)

        # 設置邊
        workflow.add_edge(START, "retrieve")  # 使用 START 常量
        workflow.add_edge("retrieve", "analyze_question")
        workflow.add_edge("analyze_question", "check_guardrail")
        workflow.add_edge("check_guardrail", END)

        self.app = workflow.compile()

    def process_query(self, query: str):
        state = {
            "messages": [HumanMessage(content=query)],
            "quert_documents": [],
            "suggestions": {},
            "dataset_name": "",
            "intention": "",
            "target_dataset": "",
            "is_allowed": True,
            "should_terminate": False,
            "response": ""
        }
        
        # 運行工作流
        result = self.app.invoke(state)
        # print(f"響應: {result['response']}\n")
        return result


if __name__ == "__main__":
    # 測試查詢
    test_queries = [
        "客家桐花在幾月的時候開？",  # first
        "請問2022桐花季經典45條桐花小旅行遊程有哪些路線和特色活動？"
    ]
    
    rag_guardrail = RAGGuardrail()

    for query in test_queries:
        print(f"\n查詢: {query}")
        state = rag_guardrail.process_query(query)