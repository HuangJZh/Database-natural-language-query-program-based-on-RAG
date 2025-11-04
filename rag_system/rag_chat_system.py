# enhanced_rag_dialogue_system.py
import os
import mysql.connector
import torch
import pandas as pd
from typing import List, Dict, Any, Tuple
import re

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import BitsAndBytesConfig


class DatabaseKnowledgeExtractor:
    def __init__(self, db_config, model_path):
        self.db_config = db_config
        self.model_path = model_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """连接数据库"""
        self.conn = mysql.connector.connect(**self.db_config)
        self.cursor = self.conn.cursor(dictionary=True)
        print("✅ 数据库连接成功")
    
    def extract_schema_info(self):
        """提取数据库模式信息"""
        schema_info = {
            "tables": {},
            "relationships": [],
            "statistics": {}
        }
        
        # 获取所有表
        self.cursor.execute("SHOW TABLES")
        tables = [list(table.values())[0] for table in self.cursor.fetchall()]
        
        for table in tables:
            # 获取表结构
            self.cursor.execute(f"DESCRIBE {table}")
            columns = self.cursor.fetchall()
            
            # 获取索引信息
            self.cursor.execute(f"SHOW INDEX FROM {table}")
            indexes = self.cursor.fetchall()
            
            schema_info["tables"][table] = {
                "columns": columns,
                "indexes": indexes
            }
            
            # 获取表统计信息
            self.cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            count_result = self.cursor.fetchone()
            schema_info["statistics"][table] = {
                "row_count": count_result["count"]
            }
        
        # 提取外键关系（简化版本）
        schema_info["relationships"] = self._extract_relationships()
        
        return schema_info
    
    def _extract_relationships(self):
        """提取表之间的关系"""
        relationships = []
        
        # 基于命名约定和数据分析推断关系
        table_pairs = [
            ("users", "orders", "user_id"),
            ("products", "order_items", "product_id"),
            ("orders", "order_items", "order_id"),
            ("categories", "products", "category_id")
        ]
        
        for table1, table2, key in table_pairs:
            relationships.append({
                "table1": table1,
                "table2": table2,
                "relationship": f"{table1}.{key} = {table2}.{key}",
                "type": "foreign_key"
            })
        
        return relationships
    
    def extract_sample_data(self, sample_size=5):
        """提取样本数据用于理解数据分布"""
        sample_data = {}
        
        tables = ["users", "products", "orders", "order_items", "categories"]
        
        for table in tables:
            try:
                self.cursor.execute(f"SELECT * FROM {table} LIMIT {sample_size}")
                sample_data[table] = self.cursor.fetchall()
            except:
                print(f"无法获取表 {table} 的样本数据")
        
        return sample_data


class AdvancedRAGSystem:
    def __init__(self, config, db_config):
        self.config = config
        self.db_config = db_config
        self.embeddings = self._init_embeddings()
        self.llm = self._init_llm()
        self.vector_db = None
        self.qa_chain = None
        
    def _init_embeddings(self):
        print("加载bge-small-zh-v1.5嵌入模型...")
        return HuggingFaceBgeEmbeddings(
            model_name=self.config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True},
            query_instruction="为这个句子生成表示以用于检索相关文章："
        )

    def _init_llm(self):
        print(f"加载Qwen3模型: {self.config.LLM_MODEL_NAME}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.LLM_MODEL_NAME)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            self.config.LLM_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="cuda",
            dtype=torch.float16,
            trust_remote_code=True
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=1000,
            temperature=0.1,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        return HuggingFacePipeline(pipeline=pipe)
    
    def load_rag_system(self):
        """加载RAG系统"""
        if os.path.exists(self.config.VECTOR_DB_DIR):
            print("加载已有向量库...")
            self.vector_db = Chroma(
                persist_directory=self.config.VECTOR_DB_DIR,
                embedding_function=self.embeddings
            )
            
            # SQL生成提示模板
            sql_prompt_template = """你是一个SQL专家。基于以下数据库结构知识和用户问题，生成准确且优化的SQL查询语句。

数据库结构信息:
{context}

用户问题: {question}

请遵循以下规则:
1. 只返回SQL查询语句，不要包含其他解释
2. 使用正确的表名和列名
3. 包含适当的WHERE条件、JOIN条件和GROUP BY子句
4. 对于分页查询使用LIMIT
5. 确保SQL语法正确

SQL查询:"""
            
            sql_prompt = PromptTemplate(
                template=sql_prompt_template,
                input_variables=["context", "question"]
            )
            
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_db.as_retriever(search_kwargs={"k": self.config.TOP_K}),
                chain_type_kwargs={"prompt": sql_prompt},
                return_source_documents=True
            )
            return True
        return False
    
    def generate_sql_without_rag(self, question: str) -> str:
        """不使用RAG生成SQL（基础版本）"""
        prompt = f"""请为以下问题生成SQL查询语句。数据库包含users, products, orders, order_items, categories等表。

问题: {question}

请只返回SQL查询语句:"""
        
        result = self.llm.invoke(prompt)
        return self._extract_sql_from_response(result)
    
    def generate_sql_with_rag(self, question: str) -> Tuple[str, List]:
        """使用RAG生成SQL"""
        result = self.qa_chain.invoke({"query": question}) 
        sql = self._extract_sql_from_response(result["result"])
        return sql, result["source_documents"]
    
    def _extract_sql_from_response(self, response: str) -> str:
        """从响应中提取SQL语句"""
        # 查找SQL开始
        sql_start = -1
        sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]
        
        for keyword in sql_keywords:
            idx = response.upper().find(keyword)
            if idx != -1 and (sql_start == -1 or idx < sql_start):
                sql_start = idx
        
        if sql_start != -1:
            # 提取到分号或结尾
            sql_end = response.find(';', sql_start)
            if sql_end == -1:
                sql_end = len(response)
            
            sql = response[sql_start:sql_end].strip()
            # 清理可能的Markdown代码块
            sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE)
            return sql.strip()
        
        return response.strip()
    
    def execute_sql_query(self, sql_query: str) -> Tuple[bool, Any]:
        """执行SQL查询并返回结果"""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(sql_query)
            result = cursor.fetchall()
            cursor.close()
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)


def generate_enhanced_overview(schema_info, sample_data):
    """生成数据库概览"""
    content = "数据库概览\n"
    content += "=" * 60 + "\n\n"
    
    content += "表统计:\n"
    for table_name, stats in schema_info["statistics"].items():
        col_count = len(schema_info["tables"][table_name]["columns"])
        content += f"- {table_name}: {col_count}列, {stats['row_count']}行数据\n"
    
    content += "\n关键关系:\n"
    for rel in schema_info["relationships"]:
        content += f"- {rel['table1']} ↔ {rel['table2']} ({rel['relationship']})\n"
    
    content += "\n数据特征:\n"
    # 添加数据分布信息
    if 'users' in sample_data:
        cities = [user['city'] for user in sample_data['users'] if 'city' in user]
        if cities:
            content += f"- 用户城市分布示例: {', '.join(set(cities))}\n"
    
    return content


def generate_detailed_table_doc(table_name, table_info, sample_data):
    """生成详细的表文档"""
    content = f"表详细文档: {table_name}\n"
    content += "=" * 60 + "\n\n"
    
    content += "列详细信息:\n"
    for col in table_info["columns"]:
        content += f"- {col['Field']}: {col['Type']} | Null: {col['Null']} | Key: {col['Key']} | Default: {col['Default']}\n"
    
    content += "\n索引信息:\n"
    for idx in table_info["indexes"]:
        content += f"- {idx['Column_name']} ({idx['Index_type']}) - {'唯一' if idx['Non_unique'] == 0 else '非唯一'}\n"
    
    if sample_data:
        content += f"\n数据示例 ({len(sample_data)} 行):\n"
        df = pd.DataFrame(sample_data)
        content += df.to_string() + "\n"
    
    return content


def generate_query_patterns():
    """生成查询模式文档"""
    content = "常用查询模式\n"
    content += "=" * 60 + "\n\n"
    
    patterns = [
        {
            "pattern": "简单单表查询",
            "description": "查询用户的基本信息",
            "sql": "SELECT user_id, username, email, city, country FROM users"
        },
        {
            "pattern": "带条件的单表查询",
            "description": "查询来自纽约的用户",
            "sql": "SELECT * FROM users WHERE city = 'New York'"
        },
        {
            "pattern": "两表连接查询", 
            "description": "查询产品及其分类信息",
            "sql": "SELECT p.product_name, p.price, c.category_name FROM products p JOIN categories c ON p.category_id = c.category_id"
        },
        {
            "pattern": "聚合查询",
            "description": "统计每个城市的用户数",
            "sql": "SELECT city, COUNT(*) as user_count FROM users GROUP BY city ORDER BY user_count DESC"
        },
        {
            "pattern": "复杂多表连接",
            "description": "查询每个用户的订单总金额",
            "sql": "SELECT u.username, SUM(o.total_amount) as total_spent FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.user_id, u.username ORDER BY total_spent DESC"
        }
    ]
    
    for pattern in patterns:
        content += f"{pattern['pattern']}:\n"
        content += f"描述: {pattern['description']}\n"
        content += f"SQL模式: {pattern['sql']}\n\n"
    
    return content


def generate_business_logic():
    """生成业务逻辑文档"""
    content = "业务逻辑和规则\n"
    content += "=" * 60 + "\n\n"
    
    business_rules = [
        "用户忠诚度等级: Bronze(青铜) < Silver(白银) < Gold(黄金) < Platinum(白金)",
        "订单状态流转: pending(待处理) → confirmed(已确认) → shipped(已发货) → delivered(已送达)",
        "支付状态: pending(待支付) → paid(已支付) → refunded(已退款)",
        "库存预警: 当stock_quantity <= min_stock_level时触发补货提醒",
        "用户注册: registration_date记录注册时间，last_login记录最后登录时间"
    ]
    
    for rule in business_rules:
        content += f"• {rule}\n"
    
    return content


def create_enhanced_database_docs(db_config, output_dir):
    """创建数据库文档"""
    extractor = DatabaseKnowledgeExtractor(db_config, "D:/Qwen/Qwen/Qwen3-8B")
    extractor.connect()
    
    # 提取更详细的信息
    schema_info = extractor.extract_schema_info()
    sample_data = extractor.extract_sample_data(10)  # 更多样本
    
    docs = []
    
    # 1. 数据库概览
    overview = generate_enhanced_overview(schema_info, sample_data)
    docs.append(("enhanced_overview.txt", overview))
    
    # 2. 详细的表文档（包含数据类型和约束）
    for table_name in schema_info["tables"]:
        table_doc = generate_detailed_table_doc(table_name, schema_info["tables"][table_name], sample_data.get(table_name, []))
        docs.append((f"table_{table_name}_detailed.txt", table_doc))
    
    # 3. 查询模式文档
    query_patterns = generate_query_patterns()
    docs.append(("query_patterns.txt", query_patterns))
    
    # 4. 业务逻辑文档
    business_logic = generate_business_logic()
    docs.append(("business_logic.txt", business_logic))
    
    # 保存文档
    os.makedirs(output_dir, exist_ok=True)
    for filename, content in docs:
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
            f.write(content)
    
    print(f"✅ 数据库文档已生成到 {output_dir}")


class Config:
    DOCUMENTS_DIR = "D:/Qwen/Qwen3/enhanced_database_docs"
    CHUNK_SIZE = 400
    CHUNK_OVERLAP = 80
    EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"  
    LLM_MODEL_NAME = "D:/Qwen/Qwen/Qwen3-8B"  
    VECTOR_DB_DIR = "vector_db_enhanced"
    TOP_K = 3


class DatabaseDialogueSystem:
    def __init__(self, config, db_config):
        self.config = config
        self.db_config = db_config
        self.rag_system = None
        self.history = []
        
    def initialize_system(self):
        """初始化系统"""
        print("🚀 初始化数据库对话系统...")
        print("=" * 60)
        
        # 第一步：创建数据库文档（如果不存在）
        if not os.path.exists(self.config.DOCUMENTS_DIR) or not os.listdir(self.config.DOCUMENTS_DIR):
            print("📝 创建数据库文档...")
            create_enhanced_database_docs(self.db_config, self.config.DOCUMENTS_DIR)
        else:
            print("📝 数据库文档已存在，跳过创建...")
        
        # 第二步：构建RAG系统
        print("\n🔧 构建RAG系统...")
        self.rag_system = AdvancedRAGSystem(self.config, self.db_config)
        
        # 如果向量库不存在，则创建
        if not os.path.exists(self.config.VECTOR_DB_DIR):
            print("创建向量库...")
            loader = DirectoryLoader(
                self.config.DOCUMENTS_DIR,
                glob="*.txt",
                loader_cls=TextLoader,
                loader_kwargs={"encoding": "utf-8"}
            )
            documents = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                separators=["\n\n", "\n", "。", "，", "；", "、", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            
            # 创建向量库
            self.rag_system.vector_db = Chroma.from_documents(
                documents=texts,
                embedding=self.rag_system.embeddings,
                persist_directory=self.config.VECTOR_DB_DIR
            )
            self.rag_system.vector_db.persist()
        
        # 加载RAG系统
        self.rag_system.load_rag_system()
        print("✅ 系统初始化完成！")
    
    def process_query(self, question: str, use_rag: bool = True) -> Dict[str, Any]:
        """处理用户查询"""
        print(f"\n{'='*80}")
        print(f"📝 用户问题: {question}")
        
        response = {
            "question": question,
            "sql_query": "",
            "execution_success": False,
            "execution_result": None,
            "error_message": "",
            "source_documents": [],
            "use_rag": use_rag
        }
        
        try:
            # 生成SQL
            if use_rag:
                sql_query, source_docs = self.rag_system.generate_sql_with_rag(question)
                response["source_documents"] = source_docs
            else:
                sql_query = self.rag_system.generate_sql_without_rag(question)
            
            response["sql_query"] = sql_query
            print(f"🔍 生成的SQL: {sql_query}")
            
            # 执行SQL（如果是SELECT查询）
            if sql_query.strip().upper().startswith('SELECT'):
                success, result = self.rag_system.execute_sql_query(sql_query)
                response["execution_success"] = success
                if success:
                    response["execution_result"] = result
                    print(f"✅ 查询成功，返回 {len(result)} 行结果")
                else:
                    response["error_message"] = result
                    print(f"❌ 查询执行失败: {result}")
            else:
                print("⚠️  非SELECT查询，跳过执行")
            
            # 添加到历史记录
            self.history.append(response)
            
        except Exception as e:
            error_msg = f"处理查询时发生错误: {str(e)}"
            response["error_message"] = error_msg
            print(f"❌ {error_msg}")
        
        return response
    
    def display_result(self, response: Dict[str, Any]):
        """显示查询结果"""
        print(f"\n📊 查询结果:")
        print(f"使用RAG: {'是' if response['use_rag'] else '否'}")
        print(f"SQL查询: {response['sql_query']}")
        
        if response["execution_success"] and response["execution_result"]:
            result = response["execution_result"]
            if result:
                df = pd.DataFrame(result)
                print(f"\n📋 查询结果 ({len(result)} 行):")
                print(df.to_string(index=False))
            else:
                print("📋 查询结果: 无数据")
        elif response["error_message"]:
            print(f"❌ 错误: {response['error_message']}")
        
        # # 显示相关文档（如果使用RAG）
        # if response["use_rag"] and response["source_documents"]:
        #     print(f"\n📚 相关参考文档:")
        #     for i, doc in enumerate(response["source_documents"][:2], 1):
        #         content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
        #         print(f"文档 {i}: {content_preview}")
    
    def show_history(self):
        """显示查询历史"""
        print(f"\n📜 查询历史 ({len(self.history)} 条):")
        for i, item in enumerate(self.history, 1):
            status = "✅" if item["execution_success"] else "❌"
            print(f"{i}. {status} {item['question']}")
    
    def run_dialogue(self):
        """运行对话系统"""
        print("\n🎯 数据库对话系统已启动！")
        print("输入 'quit' 或 'exit' 退出系统")
        print("输入 'history' 查看历史记录")
        print("输入 'toggle' 切换RAG模式")
        print("输入 'help' 查看帮助")
        
        use_rag = True
        
        while True:
            try:
                user_input = input("\n💬 请输入您的查询: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    print("👋 再见！")
                    break
                elif user_input.lower() == 'history':
                    self.show_history()
                    continue
                elif user_input.lower() == 'toggle':
                    use_rag = not use_rag
                    print(f"🔄 RAG模式已{'开启' if use_rag else '关闭'}")
                    continue
                elif user_input.lower() == 'help':
                    self.show_help()
                    continue
                elif not user_input:
                    continue
                
                # 处理查询
                response = self.process_query(user_input, use_rag)
                self.display_result(response)
                
            except KeyboardInterrupt:
                print("\n👋 用户中断，再见！")
                break
            except Exception as e:
                print(f"❌ 系统错误: {str(e)}")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
🤖 数据库对话系统帮助

可用命令:
• 输入自然语言问题 - 系统会生成并执行相应的SQL查询
• 'quit' 或 'exit' - 退出系统
• 'history' - 查看查询历史
• 'toggle' - 切换RAG模式（开启/关闭）
• 'help' - 显示此帮助信息

示例问题:
• "查询前10个用户的基本信息"
• "统计每个城市的用户数量"
• "查询来自纽约的用户"
• "查询产品及其分类信息"
• "查询每个用户的订单总金额"

RAG模式:
• 开启时：系统会参考数据库文档生成更准确的SQL
• 关闭时：系统仅基于通用知识生成SQL
        """
        print(help_text)


def main():
    # 数据库配置
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'admin',
        'database': 'test_rag_mid'
    }
    
    config = Config()
    
    # 创建并运行对话系统
    dialogue_system = DatabaseDialogueSystem(config, db_config)
    dialogue_system.initialize_system()
    dialogue_system.run_dialogue()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()