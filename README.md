# LangGraph RAG 與 Guardrail 實現

這個專案使用 LangGraph 實現了一個帶有 RAG（檢索增強生成）和 Guardrail 功能的對話系統。

## 功能特點

- 使用 LangGraph 構建工作流
- 實現 RAG 功能，支援文檔檢索和上下文增強
- 包含 Guardrail 檢查機制
- 使用 Chroma 作為向量存儲
- 支援自定義文檔添加

## 安裝

1. 克隆專案後，安裝依賴：
```bash
pip install -r requirements.txt
```

2. 創建 `.env` 文件並添加 OpenAI API 密鑰：
```
OPENAI_API_KEY=your_api_key_here
```

## 使用方法

1. 運行示例代碼：
```bash
python rag_guardrail.py
```

2. 添加自定義文檔：
```python
from rag_guardrail import add_documents

documents = [
    "你的文檔內容1",
    "你的文檔內容2"
]
add_documents(documents)
```

3. 處理查詢：
```python
from rag_guardrail import process_query

response = process_query("你的問題")
print(response)
```

## 工作流程

1. 接收用戶查詢
2. 檢索相關文檔
3. 進行 Guardrail 檢查
4. 生成響應

## 自定義 Guardrail

可以在 `check_guardrail` 函數中添加自定義的檢查規則，例如：
- 內容審核
- 敏感信息過濾
- 輸出格式驗證
- 特定規則檢查 