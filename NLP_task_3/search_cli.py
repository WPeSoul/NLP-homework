import faiss
from sentence_transformers import SentenceTransformer
import pickle
import argparse

def load_data():
    """加载预处理后的DataFrame和FAISS索引"""
    df_path = 'preprocessed_df.pkl'
    embeddings_path = 'all_embeddings.npy'
    index_path = 'faiss_index.index'

    # 加载预处理后的DataFrame Load the preprocessed DataFrame from a pickle file
    with open(df_path, 'rb') as f:
        df = pickle.load(f)

    # 加载FAISS索引 Load the FAISS index
    index = faiss.read_index(index_path)
    
    return df, index

def search(query, top_k):
    """执行搜索操作"""
    global embedder  # 使用全局变量embedder来避免重复加载模型 (Use a global variable embedder to avoid reloading the model)
    
    query_embedding = embedder.encode([query], convert_to_tensor=True).cpu().numpy() # 将查询文本编码为嵌入向量 (Encode the query text into an embedding vector)

    # 如果top_k是None，则搜索所有文档 (If top_k is None, search all documents)
    if top_k is None:
        top_k = index.ntotal  # 获取索引中的文档总数 (Get the total number of documents in the index)

    D, I = index.search(query_embedding, min(top_k, index.ntotal))  # 查询最相似的文档或所有可用文档 (Search for the most similar documents or all available documents)
    
    # 后过滤示例：这里可以根据实际情况添加更多的过滤逻辑 (Post-filtering example: add more filtering logic as needed)
    filtered_results = []
    for i in range(min(top_k, len(I[0]))):  # 确保不会超出索引范围 (Ensure not to exceed the index bounds)
        score = D[0][i]
        idx = I[0][i]
        title = df.iloc[idx]['title']
        content = df.iloc[idx]['content']
        filtered_results.append((score, title, content))
    
    return filtered_results

def get_query_from_user():
    """从用户获取查询字符串"""
    # 提示用户输入查询文本 (Prompt the user to enter the query text)
    query = input("Please enter your query: ")
    return query

def get_top_k_from_user():
    """从用户获取top_k值"""
    while True:
        try:
            # 提示用户输入返回结果的数量，留空则返回所有文档 (Prompt the user to enter the number of results or leave blank for all documents)
            top_k_input = input("Enter the number of results to return (leave blank to return all documents): ").strip()
            if top_k_input == '':
                return None  # 返回None表示查询所有文档 (Return None to indicate returning all documents)
            else:
                top_k = int(top_k_input)
                if top_k > 0:
                    return top_k  # 返回用户指定的正整数 (Return the specified positive integer)
                else:
                    print("Please enter a positive integer.")  # 提示用户输入一个正整数 (Prompt the user to enter a positive integer)
        except ValueError:
            print("Invalid input, please enter a positive integer.")  # 处理无效输入，提示用户重新输入 (Handle invalid input and prompt the user to re-enter)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    parser.add_argument("--query", type=str, help="The query string to search.")
    parser.add_argument("--top_k", type=int, help="Number of top results to return. Leave empty to return all documents.")

    args = parser.parse_args()

    if not args.query:
        print("No query provided via command line. Prompting for user input...")
        args.query = get_query_from_user()

    # 处理top_k参数，优先使用命令行参数，如果没有则提示用户输入
    # Параметр top_k обрабатывается, при этом предпочтение отдается параметру командной строки, а в случае его отсутствия пользователю предлагается ввести его
    if args.top_k is None:
        args.top_k = get_top_k_from_user()

    print("Loading data and initializing model...")
    global df, index, embedder
    df, index = load_data()
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    print(f"Searching for \"{args.query}\"...")
    results = search(args.query, args.top_k)

    if not results:
        print("No results found.")
    else:
        print("Search results:")
        for score, title, content in results:
            print(f"\nScore: {score:.4f}\nTitle: {title}\nContent: {content[:200]}...\n")
            # 输出相似度分数，文档标题，文档内容 Вывод баллов сходства, название документа, содержание документа

if __name__ == "__main__":
    main()