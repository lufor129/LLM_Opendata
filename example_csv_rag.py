from csv_to_rag import CSVToRAG
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

def main():
    # 初始化 CSVToRAG
    csv_to_rag = CSVToRAG(persist_directory="chroma_db")
    
    # 处理 CSV 文件
    csv_file = "Opendata_CSV.csv"
    if os.path.exists(csv_file):
        # 基本处理
        print("开始处理CSV文件...")
        csv_to_rag.process_csv(csv_file)
        
        # 示例查询
        print("\n执行示例查询...")
        queries = [
            "请告诉我数据中的主要类别",
            "查找最新的记录",
            "统计不同类别的数量"
        ]
        
        for query in queries:
            print(f"\n查询: {query}")
            results = csv_to_rag.query(query, k=3)
            for i, doc in enumerate(results, 1):
                print(f"\n结果 {i}:")
                print(f"内容: {doc.page_content}")
                print(f"元数据: {doc.metadata}")
    else:
        print(f"错误: 找不到文件 {csv_file}")

if __name__ == "__main__":
    main() 