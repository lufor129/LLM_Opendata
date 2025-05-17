import pandas as pd
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 加載環境變量
load_dotenv()

class CSVToRAG:
    def __init__(self, persist_directory: str = "chroma_db"):
        """初始化 CSVToRAG 類
        
        Args:
            persist_directory: Chroma數據庫的持久化目錄
        """
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _format_row_to_text(self, row: pd.Series) -> str:
        """將 DataFrame 的一行轉換為文本格式
        
        Args:
            row: pandas Series 對象，代表一行數據
            
        Returns:
            str: 格式化後的文本
        """
        formatted_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
        return formatted_text

    def _create_documents(self, df: pd.DataFrame) -> List[Document]:
        """將 DataFrame 轉換為 Document 列表
        
        Args:
            df: pandas DataFrame 對象
            
        Returns:
            List[Document]: Document 對象列表
        """
        documents = []
        for _, row in df.iterrows():
            text = self._format_row_to_text(row)
            doc = Document(
                page_content=text,
                metadata={"row_index": _}
            )
            documents.append(doc)
        return documents

    def process_csv(self, file_path: str, encoding: str = 'utf-8', batch_size: int = 100) -> None:
        """處理 CSV 文件並將內容添加到向量存儲中
        
        Args:
            file_path: CSV 文件路徑
            encoding: 文件編碼，默認為 utf-8
            batch_size: 每批處理的記錄數
        """
        try:
            # 讀取 CSV 文件，處理引號和轉義字符
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                quoting=1,  # QUOTE_ALL
                escapechar='\\',
                on_bad_lines='skip'  # 跳過有問題的行
            )
            
            # 分批處理數據
            total_rows = len(df)
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                # 將當前批次轉換為 Document 列表
                documents = self._create_documents(batch_df)
                
                # 將文檔添加到向量存儲
                self.vectorstore.add_documents(documents)
                
                # 持久化向量存儲
                self.vectorstore.persist()
                
                print(f"已處理 {end_idx}/{total_rows} 條記錄")
            
            print(f"成功處理 CSV 文件: {file_path}")
            print(f"共處理 {total_rows} 條記錄")
            
        except Exception as e:
            print(f"處理 CSV 文件時出錯: {str(e)}")

    def process_csv_with_metadata(self, file_path: str, metadata_columns: List[str], encoding: str = 'utf-8', batch_size: int = 100) -> None:
        """處理 CSV 文件並將內容添加到向量存儲中，包含額外的元數據
        
        Args:
            file_path: CSV 文件路徑
            metadata_columns: 要作為元數據的列名列表
            encoding: 文件編碼，默認為 utf-8
            batch_size: 每批處理的記錄數
        """
        try:
            # 讀取 CSV 文件，處理引號和轉義字符
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                quoting=1,  # QUOTE_ALL
                escapechar='\\',
                on_bad_lines='skip'  # 跳過有問題的行
            )
            
            # 分批處理數據
            total_rows = len(df)
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = df.iloc[start_idx:end_idx]
                
                documents = []
                for _, row in batch_df.iterrows():
                    # 提取元數據
                    metadata = {col: row[col] for col in metadata_columns if col in row}
                    
                    # 將其他列作為內容
                    content_columns = [col for col in df.columns if col in metadata_columns]
                    text = ", ".join([f"{col}: {row[col]}" for col in content_columns])
                    
                    # 創建 Document 對象
                    doc = Document(
                        page_content=text,
                        metadata=metadata
                    )
                    documents.append(doc)
                
                # 將文檔添加到向量存儲
                self.vectorstore.add_documents(documents)
                
                # 持久化向量存儲
                self.vectorstore.persist()
                
                print(f"已處理 {end_idx}/{total_rows} 條記錄")
            
            print(f"成功處理 CSV 文件: {file_path}")
            print(f"共處理 {total_rows} 條記錄")
            print(f"元數據列: {metadata_columns}")
            
        except Exception as e:
            print(f"處理 CSV 文件時出錯: {str(e)}")

    def query(self, query_text: str, k: int = 5) -> List[Document]:
        """查詢向量存儲
        
        Args:
            query_text: 查詢文本
            k: 返回的文檔數量
            
        Returns:
            List[Document]: 相關文檔列表
        """
        return self.vectorstore.similarity_search(query_text, k=k)

def main():
    # 使用示例
    csv_to_rag = CSVToRAG()
    
    # 示例 2：使用元數據
    metadata_columns = ["資料集名稱", "資料下載網址", "資料集描述", "主要欄位說明", "服務分類"]  # 根據實際CSV文件的列名修改
    csv_to_rag.process_csv_with_metadata(
        "Opendata_CSV_test.csv",
        metadata_columns=metadata_columns,
        encoding='utf-8',
        batch_size=100
    )
    print("開始查詢")
    # 示例 3：查詢
    results = csv_to_rag.query("客家桐花")
    for doc in results:
        print(f"內容: {doc.page_content}")
        print(f"元數據: {doc.metadata}")
        print("---")

if __name__ == "__main__":
    main() 