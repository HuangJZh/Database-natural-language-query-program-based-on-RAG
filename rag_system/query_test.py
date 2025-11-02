import mysql.connector
import pandas as pd

def run_sql_query(sql_query, description="自定义查询"):
    """
    执行SQL查询并显示结果
    
    参数:
        sql_query: SQL查询语句
        description: 查询描述
    """
    # 数据库配置
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'admin',
        'database': 'test_rag_mid'
    }
    
    print(f"\n🎯 {description}")
    print("=" * 60)
    print(f"SQL: {sql_query}")
    print("-" * 60)
    
    try:
        # 连接数据库
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor(dictionary=True)
        
        # 执行查询
        cursor.execute(sql_query)
        results = cursor.fetchall()
        
        if results:
            df = pd.DataFrame(results)
            print(f"✅ 返回 {len(df)} 行数据")
            print("\n📊 结果:")
            print(df.to_string(index=False))
        else:
            print("ℹ️  查询成功，但未返回数据")
            
        # 关闭连接
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ 查询失败: {e}")


if __name__ == "__main__":
    print("test_rag_mid 简单查询工具")
    
    run_sql_query("""
SELECT
    u.user_id,
    COUNT(o.order_id) AS order_count
FROM
    users u
JOIN
    orders o ON u.user_id = o.user_id
GROUP BY
    u.user_id
HAVING
    SUM(oi.item_price * oi.quantity) > (
        SELECT AVG(total_spent)
        FROM (
            SELECT
                SUM(oi.item_price * oi.quantity) AS total_spent
            FROM
                users u2
            JOIN
                orders o2 ON u2.user_id = o2.user_id
            JOIN
                order_items oi ON o2.order_id = oi.order_id
            GROUP BY
                u2.user_id
        ) AS avg_subquery
    )
ORDER BY
    SUM(oi.item_price * oi.quantity) DESC
LIMIT 10
    """, "用户订单连接查询")
    
    
    print("\n💡 提示: 请编辑此文件，取消注释并修改上面的查询语句来测试您的SQL查询")