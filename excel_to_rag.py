import pandas as pd
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class ExcelToRAG:
    def __init__(self):
        """初始化 ExcelToRAG 类"""
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _format_row_to_text(self, row: pd.Series) -> str:
        """将 DataFrame 的一行转换为文本格式
        
        Args:
            row: pandas Series 对象，代表一行数据
            
        Returns:
            str: 格式化后的文本
        """
        # 将每个单元格的内容转换为字符串，并用逗号分隔
        formatted_text = ", ".join([f"{col}: {val}" for col, val in row.items()])
        return formatted_text

    def _create_documents(self, df: pd.DataFrame) -> List[Document]:
        """将 DataFrame 转换为 Document 列表
        
        Args:
            df: pandas DataFrame 对象
            
        Returns:
            List[Document]: Document 对象列表
        """
        documents = []
        for _, row in df.iterrows():
            # 将每一行转换为文本
            text = self._format_row_to_text(row)
            # 创建 Document 对象，将行号作为元数据
            doc = Document(
                page_content=text,
                metadata={"row_index": _}
            )
            documents.append(doc)
        return documents

    def process_excel(self, file_path: str, sheet_name: str = None) -> None:
        """处理 Excel 文件并将内容添加到向量存储中
        
        Args:
            file_path: Excel 文件路径
            sheet_name: 工作表名称，如果为 None 则读取第一个工作表
        """
        try:
            # 读取 Excel 文件
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # 将 DataFrame 转换为 Document 列表
            documents = self._create_documents(df)
            
            # 将文档添加到向量存储
            self.vectorstore.add_documents(documents)
            
            print(f"成功处理 Excel 文件: {file_path}")
            print(f"共处理 {len(documents)} 条记录")
            
        except Exception as e:
            print(f"处理 Excel 文件时出错: {str(e)}")

    def process_excel_with_metadata(self, file_path: str, metadata_columns: List[str], sheet_name: str = None) -> None:
        """处理 Excel 文件并将内容添加到向量存储中，包含额外的元数据
        
        Args:
            file_path: Excel 文件路径
            metadata_columns: 要作为元数据的列名列表
            sheet_name: 工作表名称，如果为 None 则读取第一个工作表
        """
        try:
            # 读取 Excel 文件
            df = pd.read_excel(file_path, sheet_name=sheet_name)
            
            documents = []
            for _, row in df.iterrows():
                # 提取元数据
                metadata = {col: row[col] for col in metadata_columns if col in row}
                metadata["row_index"] = _
                
                # 将其他列作为内容
                content_columns = [col for col in df.columns if col not in metadata_columns]
                text = ", ".join([f"{col}: {row[col]}" for col in content_columns])
                
                # 创建 Document 对象
                doc = Document(
                    page_content=text,
                    metadata=metadata
                )
                documents.append(doc)
            
            # 将文档添加到向量存储
            self.vectorstore.add_documents(documents)
            
            print(f"成功处理 Excel 文件: {file_path}")
            print(f"共处理 {len(documents)} 条记录")
            print(f"元数据列: {metadata_columns}")
            
        except Exception as e:
            print(f"处理 Excel 文件时出错: {str(e)}")

def main():
    # 使用示例
    excel_to_rag = ExcelToRAG()
    
    # 示例 1：基本使用
    excel_to_rag.process_excel("example.xlsx")
    
    # 示例 2：使用元数据
    metadata_columns = ["ID", "日期", "类别"]
    excel_to_rag.process_excel_with_metadata(
        "example.xlsx",
        metadata_columns=metadata_columns
    )

if __name__ == "__main__":
    main() 