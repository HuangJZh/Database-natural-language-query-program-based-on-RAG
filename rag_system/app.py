import os
import sys
import asyncio
import subprocess
import threading
from typing import Optional, Dict
from fastapi import FastAPI, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from datetime import datetime
import re

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from rag_system import rag_chat_system
except Exception:
    import rag_chat_system

try:
    from rag_system import database_rag_system
except Exception:
    import database_rag_system

import psutil
import torch
from typing import Any

# Shared LLM loading
_shared_llm: Any = None
_shared_llm_lock = threading.Lock()
_shared_embeddings: Any = None
_shared_embeddings_lock = threading.Lock()

# Page mode
app = FastAPI(title="基于RAG的自然语言SQL查询")
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'static')), name="static")
_active_mode: str = "chat"

# Database configuration
DEFAULT_DB_CONFIG = {
    'host': os.environ.get('RAG_DB_HOST', 'localhost'),
    'user': os.environ.get('RAG_DB_USER', 'root'),
    'password': os.environ.get('RAG_DB_PASSWORD', 'admin'),
    'database': os.environ.get('RAG_DB_NAME', 'test_rag_mid')
}

# Chat system instance
_dialogue_system: Optional[rag_chat_system.DatabaseDialogueSystem] = None
_initialized = False
_initializing = False

# Chat history management
_chat_history: list[Dict[str, Any]] = []
_chat_history_lock = threading.Lock()

# Test system instance
_test_system: Optional[rag_chat_system.DatabaseDialogueSystem] = None
_test_initialized = False
_test_initializing = False

# Python output buffer
_py_output_lines: list[str] = []
_PY_OUTPUT_MAX_LINES = 5
_py_output_lock = threading.Lock()

_test_success_stats = {
    'total_tests': 0,
    'no_rag_exec_success_count': 0,
    'with_rag_exec_success_count': 0,
    'no_rag_exec_success_rate': 0,
    'with_rag_exec_success_rate': 0
}
_test_success_stats_lock = threading.Lock()


def _append_py_output(msg: str):
    try:
        try:
            print(msg, flush=True)
        except Exception:
            pass
        with _py_output_lock:
            _py_output_lines.append(msg)
            if len(_py_output_lines) > _PY_OUTPUT_MAX_LINES:
                while len(_py_output_lines) > _PY_OUTPUT_MAX_LINES:
                    _py_output_lines.pop(0)
    except Exception:
        pass


def init_chat_sync() -> None:
    global _dialogue_system, _initialized, _initializing
    if _initialized:
        _append_py_output('对话功能已完成初始化！')
        return
    _initializing = True
    _append_py_output('对话功能初始化开始')
    try:
        cfg = rag_chat_system.Config()
        db_cfg = DEFAULT_DB_CONFIG
        with _shared_llm_lock:
            shared_llm = _shared_llm
        with _shared_embeddings_lock:
            shared_embeddings = _shared_embeddings
        ds = rag_chat_system.DatabaseDialogueSystem(
            config=cfg,
            db_config=db_cfg,
            llm=shared_llm,
            embeddings=shared_embeddings)
        ds.initialize_system()
        _dialogue_system = ds
        _initialized = True
        _append_py_output('对话功能初始化完成！')
    except Exception as e:
        _append_py_output(f'对话功能初始化失败: {e}')
        raise
    finally:
        _initializing = False


def init_test_sync() -> None:
    global _test_system, _test_initialized, _test_initializing
    if _test_initialized:
        _append_py_output('测试功能已完成初始化！')
        return
    _test_initializing = True
    _append_py_output('测试功能初始化开始')
    try:
        cfg = rag_chat_system.Config()
        db_cfg = DEFAULT_DB_CONFIG
        with _shared_llm_lock:
            shared_llm = _shared_llm
        with _shared_embeddings_lock:
            shared_embeddings = _shared_embeddings
        ts = rag_chat_system.DatabaseDialogueSystem(
            config=cfg,
            db_config=db_cfg,
            llm=shared_llm,
            embeddings=shared_embeddings)
        ts.initialize_system()
        _test_system = ts
        _test_initialized = True
        _append_py_output('测试功能初始化完成！')
    except Exception as e:
        _append_py_output(f'测试功能初始化失败: {e}')
        raise
    finally:
        _test_initializing = False


class QueryRequest(BaseModel):
    question: str
    use_rag: Optional[bool] = True
    hybrid: Optional[bool] = False


@app.get("/", response_class=HTMLResponse)
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'static', 'index.html'))


def _get_system_status() -> dict:
    """Return a small snapshot of CPU, memory and GPU state."""
    try:
        vm = psutil.virtual_memory()
        cpu = psutil.cpu_percent(interval=0.05)
        gpu_available = False
        gpu_count = 0
        gpus = []
        try:
            gpu_available = torch.cuda.is_available()
        except Exception:
            gpu_available = False

        if gpu_available:
            try:
                gpu_count = torch.cuda.device_count()
            except Exception:
                gpu_count = 0

        if gpu_available and gpu_count > 0:
            for i in range(gpu_count):
                try:
                    props = None
                    allocated = None
                    reserved = None
                    try:
                        props = torch.cuda.get_device_properties(i)
                    except Exception:
                        props = None
                    try:
                        allocated = torch.cuda.memory_allocated(i)
                    except Exception:
                        allocated = None
                    try:
                        reserved = torch.cuda.memory_reserved(i)
                    except Exception:
                        reserved = None

                    util = None
                    # nvidia-smi may not be present in all environments; wrap it.
                    try:
                        p = subprocess.run([
                            'nvidia-smi',
                            f'--id={i}',
                            '--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used',
                            '--format=csv,noheader,nounits'
                        ], capture_output=True, text=True, check=True)
                        out = [x.strip() for x in p.stdout.strip().split(',')]
                        if len(out) >= 4:
                            util = {
                                'gpu': out[0],
                                'mem_util': out[1],
                                'mem_total_mb': out[2],
                                'mem_used_mb': out[3]
                            }
                    except Exception:
                        util = None

                    gpus.append({
                        'id': i,
                        'name': props.name if props is not None else None,
                        'total_memory': getattr(props, 'total_memory', None),
                        'allocated': allocated,
                        'reserved': reserved,
                        'nvidia_smi': util
                    })
                except Exception:
                    # swallow per-device exceptions and continue
                    continue

        return {
            'cpu_percent': cpu,
            'mem_total': vm.total,
            'mem_used': vm.used,
            'mem_percent': vm.percent,
            'gpu_available': gpu_available,
            'gpu_count': gpu_count,
            'gpus': gpus
        }
    except Exception as e:
        return {'error': str(e)}


@app.get("/status")
async def status():
    sys_status = _get_system_status()
    return {
        "initialized": _initialized,
        "initializing": _initializing,
        "system": sys_status
    }


@app.get('python_output')
async def python_output():
    try:
        with _py_output_lock:
            lines = list(_py_output_lines)[-_PY_OUTPUT_MAX_LINES:]
            lines = list(reversed(lines))
        return {
            'lines': lines
        }
    except Exception as e:
        return {"lines": [f"Error retrieving output: {e}"]}


@app.post("/init")
async def init_system(background_tasks: BackgroundTasks):
    global _dialogue_system, _initialized, _initializing
    if _initialized:
        return {"status": "已完成初始化！"}
    if _initializing:
        return {"status": "正在初始化..."}

    def _init_all():
        try:
            _append_py_output('全部初始化开始')
            try:
                cfg = rag_chat_system.Config()

                def _load_shared():
                    global _shared_llm
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
                        from transformers import BitsAndBytesConfig
                        from langchain_community.llms import HuggingFacePipeline

                        model_name = cfg.LLM_MODEL_NAME
                        _append_py_output(f'加载共享LLM模型: {model_name}')
                        tokenizer = AutoTokenizer.from_pretrained(model_name)
                        quantization_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16,
                            bnb_4bit_quant_type="nf4",
                            bnb_4bit_use_double_quant=True,
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=quantization_config,
                            device_map="cuda",
                            # device_map="auto",
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
                        _shared_llm = HuggingFacePipeline(pipeline=pipe)
                        _append_py_output('加载LLM...')
                    except Exception as e:
                        _append_py_output(f'加载LLM失败: {e}')

                asyncio.run(asyncio.to_thread(_load_shared))
            except Exception as e:
                _append_py_output(f'加载共享LLM失败: {e}')

            init_chat_sync()
            init_test_sync()
            _append_py_output('全部初始化完成！')
        except Exception as e:
            _append_py_output(f'全部初始化失败: {e}')
            raise

    background_tasks.add_task(_init_all)
    return {"status": "初始化已启动！请稍后检查状态。"}


@app.post('/load_embeddings')
async def load_embeddings():
    global _shared_embeddings
    with _shared_embeddings_lock:
        if _shared_embeddings is not None:
            return {"status": "嵌入模型已加载"}
        try:
            cfg = rag_chat_system.Config()
            _append_py_output(f'加载共享嵌入模型{cfg.EMBEDDING_MODEL_NAME}...')
            from langchain_community.embeddings import HuggingFaceEmbeddings
            emb = HuggingFaceEmbeddings(
                model_name=cfg.EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
                # model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                query_instruction="为这个句子生成表示以用于检索相关文章："
            )
            _shared_embeddings = emb
            _append_py_output('共享嵌入模型加载完成！')
            return {"status": "嵌入模型加载完成"}
        except Exception as e:
            _append_py_output(f'加载嵌入模型失败: {e}')
            return JSONResponse(status_code=500, content={"status": f"加载嵌入模型失败: {e}"})


@app.post('/init_test')
async def init_test_system(background_tasks: BackgroundTasks):
    global _test_system, _test_initialized, _test_initializing
    if _test_initialized:
        return {"status": "测试功能已完成初始化！"}
    if _test_initializing:
        return {"status": "测试功能正在初始化..."}
    _test_initializing = True

    def _init_test():
        global _test_system, _test_initialized, _test_initializing
        try:
            cfg = rag_chat_system.Config()
            db_cfg = DEFAULT_DB_CONFIG
            with _shared_llm_lock:
                shared_llm = _shared_llm
            with _shared_embeddings_lock:
                shared_emb = _shared_embeddings
            ts = rag_chat_system.DatabaseDialogueSystem(
                config=cfg,
                db_config=db_cfg,
                llm=shared_llm,
                embeddings=shared_emb)
            ts.initialize_system()
            _test_system = ts
            _test_initialized = True
        except Exception:
            _test_initializing = False
            raise
        finally:
            _test_initializing = False

    background_tasks.add_task(_init_test)
    return {"status": "测试功能初始化已启动！请稍后检查状态。"}


def _extract_sql_from_response(response: str) -> str:
    sql_start = -1
    sql_keywords = ["SELECT", "INSERT", "UPDATE", "DELETE", "WITH"]

    for keyword in sql_keywords:
        idx = response.upper().find(keyword)
        if idx != -1 and (sql_start == -1 or idx < sql_start):
            sql_start = idx

    if sql_start != -1:
        sql_end = response.find(';', sql_start)
        if sql_end == -1:
            sql_end = len(response)

        sql = response[sql_start:sql_end].strip()
        sql = re.sub(r'^```sql\s*|\s*```$', '', sql, flags=re.IGNORECASE)
        return sql.strip()
    return response.strip()


def _normalize_sql(s: str) -> str:
    if not s:
        return ""
    # simple normalization: uppercase, remove whitespace and trailing semicolons
    try:
        s2 = s.strip().rstrip(';')
        s2 = ' '.join(s2.split())
        return s2.upper()
    except Exception:
        return s


def _similarity(a: str, b: str) -> float:
    """A tiny token-overlap similarity (0..1)."""
    if not a or not b:
        return 0.0
    ta = set(a.split())
    tb = set(b.split())
    inter = ta.intersection(tb)
    union = ta.union(tb)
    return len(inter) / max(1, len(union))


def get_test_scenarios():
    """从database_rag_system.py获取测试场景"""
    try:
        comparator = database_rag_system.EnhancedRAGComparison(DEFAULT_DB_CONFIG, "dummy_model_path")
        return comparator.get_challenging_test_scenarios()
    except Exception as e:
        _append_py_output(f"获取测试场景失败: {e}")
        # 返回默认测试场景
        return [
            {
                "name": "场景1: 简单单表查询",
                "question": "查询前10个用户的基本信息，限制10个输出",
                "expected_sql": "SELECT user_id, username, email, city, country FROM users LIMIT 10"
            },
            {
                "name": "场景2: 带条件的单表查询",
                "question": "查询来自纽约的用户，限制10个输出",
                "expected_sql": "SELECT * FROM users WHERE city = 'New York' LIMIT 10"
            },
            {
                "name": "场景3: 两表连接查询",
                "question": "查询产品及其分类信息，限制10个输出",
                "expected_sql": "SELECT p.product_name, p.price, c.category_name FROM products p JOIN categories c ON p.category_id = c.category_id LIMIT 10"
            },
            {
                "name": "场景4: 聚合查询",
                "question": "统计每个城市的用户数量，限制10个输出",
                "expected_sql": "SELECT city, COUNT(*) as user_count FROM users GROUP BY city ORDER BY user_count DESC LIMIT 10"
            },
            {
                "name": "场景5: 复杂多表连接",
                "question": "查询每个用户的订单总金额，限制10个输出",
                "expected_sql": "SELECT u.username, SUM(o.total_amount) as total_spent FROM users u JOIN orders o ON u.user_id = o.user_id GROUP BY u.user_id, u.username ORDER BY total_spent DESC LIMIT 10"
            },
            {
                "name": "场景6: 复杂列名查询",
                "question": "查询用户的注册日期和最后登录时间，显示用户ID、用户名和城市，限制10个输出",
                "expected_sql": "SELECT user_id, username, city, registration_date, last_login FROM users LIMIT 10"
            },
            {
                "name": "场景7: 多表连接与特定列",
                "question": "查询订单详情，包括订单ID、用户名、产品名称、数量和单价，限制10个输出",
                "expected_sql": """SELECT o.order_id, u.username, p.product_name, oi.quantity, oi.unit_price
                                   FROM orders o
                                            JOIN users u ON o.user_id = u.user_id
                                            JOIN order_items oi ON o.order_id = oi.order_id
                                            JOIN products p ON oi.product_id = p.product_id LIMIT 10"""
            },
            {
                "name": "场景8: 聚合函数与分组",
                "question": "统计每个产品类别的平均价格和产品数量，按平均价格降序排列，限制10个输出",
                "expected_sql": """SELECT c.category_name,
                                          AVG(p.price)        as avg_price,
                                          COUNT(p.product_id) as product_count
                                   FROM products p
                                            JOIN categories c ON p.category_id = c.category_id
                                   GROUP BY c.category_id, c.category_name
                                   ORDER BY avg_price DESC"""
            },
            {
                "name": "场景9: 复杂条件查询",
                "question": "查询最近30天内注册且来自美国纽约的黄金等级用户，限制10个输出",
                "expected_sql": """SELECT user_id, username, email, city, loyalty_level, registration_date
                                   FROM users
                                   WHERE city = 'New York'
                                     AND country = 'USA'
                                     AND loyalty_level = 'Gold'
                                     AND registration_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)"""
            },
            {
                "name": "场景10: 子查询与高级分析",
                "question": "查询消费金额高于平均消费水平的用户及其订单总数，限制10个输出",
                "expected_sql": """SELECT u.username,
                                          COUNT(o.order_id)   as order_count,
                                          SUM(o.total_amount) as total_spent
                                   FROM users u
                                            JOIN orders o ON u.user_id = o.user_id
                                   GROUP BY u.user_id, u.username
                                   HAVING total_spent > (SELECT AVG(total_amount) FROM orders)
                                   ORDER BY total_spent DESC LIMIT 10"""
            }
        ]


_test_run_lock = threading.Lock()
_test_running = False
_last_test_results = None
_progress_results: list[Dict[str, Any]] = []
_progress_lock = threading.Lock()
_test_stop_requested = False


# 在 app.py 中找到 _run_tests_sync 函数，修改如下：

def _run_tests_sync():
    global _test_running, _last_test_results, _test_stop_requested, _test_success_stats
    results = []
    
    # 重置统计信息
    with _test_success_stats_lock:
        _test_success_stats = {
            'total_tests': 0,
            'no_rag_exec_success_count': 0,
            'with_rag_exec_success_count': 0,
            'no_rag_exec_success_rate': 0,
            'with_rag_exec_success_rate': 0
        }
    
    try:
        scenarios = get_test_scenarios()
        total_scenarios = len(scenarios)
        
        # 更新总测试数
        with _test_success_stats_lock:
            _test_success_stats['total_tests'] = total_scenarios
        
        _append_py_output(f'✅ 获取到 {total_scenarios} 个测试场景')

        for idx, sc in enumerate(scenarios, start=1):
            if _test_stop_requested:
                _append_py_output('⚠️ 测试运行被请求停止')
                break
            q = sc.get('question', '')
            expected = sc.get('expected_sql', '')

            _append_py_output(f'➡️ 测试场景 {idx}/{total_scenarios}: {q}')

            # 使用rag_chat_system的方法生成SQL
            sql_no_rag = ""
            sql_with_rag = ""
            source_docs = []

            try:
                # 无RAG生成
                if _test_system and _test_system.rag_system:
                    sql_no_rag = _test_system.rag_system.generate_sql_without_rag(q)
            except Exception as e:
                sql_no_rag = f"Error generating SQL without RAG: {e}"

            try:
                # 有RAG生成
                if _test_system and _test_system.rag_system:
                    sql_with_rag, source_docs = _test_system.rag_system.generate_sql_with_rag(q)
            except Exception as e:
                sql_with_rag = f"Error generating SQL with RAG: {e}"
                source_docs = []

            # 执行SQL并获取结果
            no_rag_exec = {'executed': False, 'success': False, 'error': None, 'result': None}
            with_rag_exec = {'executed': False, 'success': False, 'error': None, 'result': None}

            # 执行无RAG的SQL
            try:
                if isinstance(sql_no_rag, str) and sql_no_rag.upper().startswith('SELECT'):
                    no_rag_exec['executed'] = True
                    if _test_system and _test_system.rag_system:
                        success, result = _test_system.rag_system.execute_sql_query(sql_no_rag)
                        no_rag_exec['success'] = bool(success)
                        if success:
                            no_rag_exec['result'] = result
                        else:
                            no_rag_exec['error'] = result
            except Exception as e:
                no_rag_exec['error'] = str(e)

            # 执行有RAG的SQL
            try:
                if isinstance(sql_with_rag, str) and sql_with_rag.upper().startswith('SELECT'):
                    with_rag_exec['executed'] = True
                    if _test_system and _test_system.rag_system:
                        success, result = _test_system.rag_system.execute_sql_query(sql_with_rag)
                        with_rag_exec['success'] = bool(success)
                        if success:
                            with_rag_exec['result'] = result
                        else:
                            with_rag_exec['error'] = result
            except Exception as e:
                with_rag_exec['error'] = str(e)

            # 更新实时统计信息
            with _test_success_stats_lock:
                if no_rag_exec.get('executed', False) and no_rag_exec.get('success', False):
                    _test_success_stats['no_rag_exec_success_count'] += 1
                
                if with_rag_exec.get('executed', False) and with_rag_exec.get('success', False):
                    _test_success_stats['with_rag_exec_success_count'] += 1
                
                # 计算实时成功率
                if _test_success_stats['total_tests'] > 0:
                    _test_success_stats['no_rag_exec_success_rate'] = round(
                        (_test_success_stats['no_rag_exec_success_count'] / idx) * 100, 1
                    )
                    _test_success_stats['with_rag_exec_success_rate'] = round(
                        (_test_success_stats['with_rag_exec_success_count'] / idx) * 100, 1
                    )
            
            entry = {
                'name': sc.get('name', ''),
                'question': q,
                'expected_sql': expected,
                'sql_no_rag': sql_no_rag,
                'sql_with_rag': sql_with_rag,
                'source_documents': [getattr(d, 'page_content', str(d)) for d in (source_docs or [])],
                'no_rag_exec': no_rag_exec,
                'with_rag_exec': with_rag_exec
            }
            results.append(entry)
            
            try:
                ser = _serialize_response(entry)
            except Exception:
                ser = entry
                
            with _progress_lock:
                _progress_results.append(ser)
            
            # 实时输出统计信息
            with _test_success_stats_lock:
                _append_py_output(
                    f"📊 实时统计 [{idx}/{total_scenarios}] - "
                    f"无RAG: {_test_success_stats['no_rag_exec_success_rate']}% "
                    f"有RAG: {_test_success_stats['with_rag_exec_success_rate']}%"
                )
        
        _last_test_results = {"results": results}
        
        # 最终统计
        with _test_success_stats_lock:
            final_stats = _test_success_stats.copy()
        _append_py_output(
            f"🎯 最终统计 - "
            f"无RAG成功率: {final_stats['no_rag_exec_success_rate']}%, "
            f"有RAG成功率: {final_stats['with_rag_exec_success_rate']}%"
        )
        
    except Exception as e:
        _last_test_results = {"error": f"测试运行失败: {e}"}
        _append_py_output(f"❌ 测试运行失败: {e}")
    finally:
        with _test_run_lock:
            _test_running = False
        _test_stop_requested = False

# 在 app.py 中添加计算成功率的函数
def calculate_success_stats():
    """计算执行成功率统计"""
    if _last_test_results is None or 'results' not in _last_test_results:
        return {
            'total_tests': 0,
            'no_rag_exec_success_rate': 0,
            'with_rag_exec_success_rate': 0
        }
    
    results = _last_test_results['results']
    total_tests = len(results)
    
    if total_tests == 0:
        return {
            'total_tests': 0,
            'no_rag_exec_success_rate': 0,
            'with_rag_exec_success_rate': 0
        }
    
    # 计算执行成功率
    no_rag_success_count = 0
    with_rag_success_count = 0
    
    for result in results:
        no_rag_exec = result.get('no_rag_exec', {})
        with_rag_exec = result.get('with_rag_exec', {})
        
        if no_rag_exec.get('executed', False) and no_rag_exec.get('success', False):
            no_rag_success_count += 1
            
        if with_rag_exec.get('executed', False) and with_rag_exec.get('success', False):
            with_rag_success_count += 1
    
    no_rag_exec_success_rate = round((no_rag_success_count / total_tests) * 100, 1) if total_tests > 0 else 0
    with_rag_exec_success_rate = round((with_rag_success_count / total_tests) * 100, 1) if total_tests > 0 else 0
    
    return {
        'total_tests': total_tests,
        'no_rag_exec_success_rate': no_rag_exec_success_rate,
        'with_rag_exec_success_rate': with_rag_exec_success_rate
    }

# 在 app.py 中添加新的路由来获取统计信息
@app.get("/success_stats")
async def success_stats():
    """获取执行成功率统计（支持实时统计）"""
    try:
        # 如果测试正在运行，返回实时统计
        if _test_running:
            with _test_success_stats_lock:
                stats = _test_success_stats.copy()
            return stats
        else:
            # 测试不在运行时，使用原有逻辑
            stats = calculate_success_stats()
            return stats
    except Exception as e:
        _append_py_output(f"❌ 获取统计信息失败: {e}")
        return {
            'total_tests': 0,
            'no_rag_exec_success_rate': 0,
            'with_rag_exec_success_rate': 0
        }


@app.get('/run_tests_status')
async def run_tests_status():
    return {"running": _test_running}


@app.post('/run_tests_start')
async def run_tests_start(background: BackgroundTasks):
    global _test_running
    if not _test_initialized or _test_system is None:
        return JSONResponse(status_code=400, content={"error": "测试功能未初始化"})

    with _test_run_lock:
        if _test_running:
            return {"status": "测试运行已在进行中"}
        _test_running = True
    global _progress_results, _last_test_results, _test_stop_requested
    with _progress_lock:
        _progress_results = []
    _last_test_results = None
    _test_stop_requested = False
    t = threading.Thread(target=_run_tests_sync, daemon=True)
    t.start()
    return {"status": "测试运行已启动"}


@app.post('/run_tests_stop')
async def run_tests_stop():
    """Request the background test run to stop gracefully."""
    global _test_stop_requested
    with _test_run_lock:
        if not _test_running:
            return {"status": "not_running"}
        _test_stop_requested = True
    return {"status": "stop_requested"}


@app.get("/run_tests_progress")
async def run_tests_progress():
    with _progress_lock:
        out = list(_progress_results)
    return {"running": _test_running, "results": out}


@app.get("/run_tests_result")
async def run_tests_result():
    if _last_test_results is None:
        return JSONResponse(status_code=400, content={"error": "没有可用的测试结果"})
    return _last_test_results


@app.post('/switch_mode')
async def switch_mode(mode: str):
    global _active_mode
    if mode not in ['chat', 'test']:
        return JSONResponse(status_code=400, content={"error": "无效的模式"})
    if mode == _active_mode:
        return {"status": f"已处于模式: {mode}", "activate_mode": _active_mode
                }

    try:
        if mode == 'chat':
            with _shared_embeddings_lock:
                shared_embeddings = _shared_embeddings
            if _dialogue_system and getattr(_dialogue_system, 'rag_system', None):
                _dialogue_system.rag_system.embeddings = shared_embeddings
            else:
                _dialogue_system.rag_system.ensure_embeddings()
            if _test_system and getattr(_test_system, 'rag_system', None):
                _test_system.rag_system.unload_embeddings()
        else:
            with _shared_embeddings_lock:
                shared_embeddings = _shared_embeddings
            if _test_system and getattr(_test_system, 'rag_system', None):
                if shared_embeddings is not None:
                    _test_system.rag_system.embeddings = shared_embeddings
                else:
                    _test_system.rag_system.ensure_embeddings()
            if _dialogue_system and getattr(_dialogue_system, 'rag_system', None):
                _dialogue_system.rag_system.unload_embeddings()

        _active_mode = mode
        _append_py_output(f'切换到模式: {mode}')
        return {"status": f"已切换到模式: {mode}", "activate_mode": _active_mode
                }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"切换模式失败: {e}"})


def _serialize_response(resp: Dict[str, Any]) -> Dict[str, Any]:
    out = resp.copy()
    docs = out.get('source_documents') or []
    try:
        simple_docs = []
        for d in docs:
            content = getattr(d, 'page_content', None)
            metadata = getattr(d, 'metadata', None)
            if content is None and isinstance(d, dict):
                content = d.get('page_content')
                metadata = d.get('metadata')
            simple_docs.append({
                'page_content': (content[:1000] + '...') if isinstance(content, str) and len(
                    content) > 1000 else content,
                'metadata': metadata
            })
        out['source_documents'] = simple_docs
    except Exception:
        out['source_documents'] = []

    exec_res = out.get('execution_result')
    if exec_res is None:
        pass
    else:
        try:
            out['source_documents'] = exec_res
        except Exception:
            out['source_documents'] = str(exec_res)
    return out


@app.post("/query")
async def query(request: QueryRequest):
    global _dialogue_system, _initialized
    if not _initialized or _dialogue_system is None:
        return JSONResponse(status_code=400, content={"status": "对话功能未初始化"})

    def _run():
        _append_py_output(f"收到问题: {request.question}")
        q = request.question
        timestamp = datetime.utcnow().isoformat() + 'Z'

        if bool(request.hybrid):
            try:
                if getattr(_dialogue_system, 'rag_system', None):
                    try:
                        _dialogue_system.rag_system.ensure_llm()
                    except Exception:
                        pass
                    try:
                        _dialogue_system.rag_system.ensure_embeddings()
                    except Exception:
                        pass

                sql_no_rag = _dialogue_system.rag_system.generate_sql_without_rag(q)
            except Exception as e:
                sql_no_rag = f"Error generating SQL without RAG: {e}"

            try:
                sql_with_rag, src = _dialogue_system.rag_system.generate_sql_with_rag(q)
            except Exception as e:
                sql_with_rag = f"Error generating SQL with RAG: {e}"
                src = []

            no_rag_exec = {
                'executed': False, 'success': False, 'error': None, 'result': None}
            with_rag_exec = {
                'executed': False, 'success': False, 'error': None, 'result': None
            }

            try:
                if isinstance(sql_no_rag, str) and sql_no_rag.upper().startswith('SELECT'):
                    no_rag_exec['executed'] = True
                    ok, res = _dialogue_system.rag_system.execute_sql_query(sql_no_rag)
                    no_rag_exec['success'] = bool(ok)
                    if ok:
                        no_rag_exec['result'] = res
                    else:
                        no_rag_exec['error'] = res
            except Exception as e:
                no_rag_exec['error'] = str(e)

            try:
                if isinstance(sql_with_rag, str) and sql_with_rag.upper().startswith('SELECT'):
                    with_rag_exec['executed'] = True
                    ok, res = _dialogue_system.rag_system.execute_sql_query(sql_with_rag)
                    with_rag_exec['success'] = bool(ok)
                    if ok:
                        with_rag_exec['result'] = res
                    else:
                        with_rag_exec['error'] = res
            except Exception as e:
                with_rag_exec['error'] = str(e)

            entry = {
                'timestamp': timestamp,
                'question': q,
                'mode': 'hybrid',
                'sql_no_rag': sql_no_rag,
                'sql_with_rag': sql_with_rag,
                'no_rag_exec': no_rag_exec,
                'with_rag_exec': with_rag_exec,
                'source_documents': [getattr(d, 'page_content', str(d)) for d in (src or [])]
            }
            with _chat_history_lock:
                _chat_history.append(entry)

            return {
                'question': q,
                'sql_query_no_rag': sql_no_rag,
                'sql_query_with_rag': sql_with_rag,
                'no_rag_execution': no_rag_exec,
                'with_rag_execution': with_rag_exec,
                'source_documents': src
            }

        try:
            if request.use_rag and _dialogue_system and getattr(_dialogue_system, 'rag_system', None):
                _dialogue_system.rag_system.ensure_llm()
                _dialogue_system.rag_system.ensure_embeddings()
        except Exception:
            pass

        res = _dialogue_system.process_query(q, bool(request.use_rag))
        hist_item = {
            'time': timestamp,
            'question': q,
            'mode': 'rag' if bool(request.use_rag) else 'no_rag',
            'sql': res.get('sql_query'),
            'execution_success': bool(res.get('execution_success')),
            'execution_result': res.get('execution_result') if res.get('execution_success') else None,
            'error_message': res.get('error_message') if not res.get('execution_success') else None,
            'source_documents': res.get('source_documents')
        }
        with _chat_history_lock:
            _chat_history.append(hist_item)

        _append_py_output(f"query: finished")
        return res

    resp = await asyncio.to_thread(_run)
    return _serialize_response(resp)


@app.get("/history")
async def history():
    global _chat_history, _dialogue_system, _initialized
    if not _initialized or _dialogue_system is None:
        return JSONResponse(status_code=400, content={"status": "对话功能未初始化"})

    with _chat_history_lock:
        out = list(_chat_history)

    safe_out = []
    for entry in out:
        try:
            saft_entry = entry.copy()
            if 'source_documents' in saft_entry and isinstance(saft_entry['source_documents'], list):
                try:
                    se = []
                    for d in saft_entry['source_documents']:
                        if isinstance(d, dict):
                            se.append(d)
                        else:
                            se.append({'page_content': str(d)})
                    saft_entry['source_documents'] = se
                except Exception:
                    saft_entry['source_documents'] = []
            safe_out.append(saft_entry)
        except Exception:
            continue
    return safe_out


if __name__ == "__main__":
    print("Starting RAG System API server...")
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning", reload=False)