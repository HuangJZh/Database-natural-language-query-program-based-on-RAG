import mysql.connector
import random
import string
from datetime import datetime, timedelta
import time

class LargeDatabaseCreator:
    def __init__(self, db_config):
        self.db_config = db_config
        self.conn = None
        self.cursor = None
    
    def connect(self):
        """连接数据库"""
        self.conn = mysql.connector.connect(**self.db_config)
        self.cursor = self.conn.cursor()
        print("✅ 数据库连接成功")
    
    def create_tables(self):
        """创建大规模业务表结构"""
        print("🗃️ 创建大规模业务表结构...")
        
        # 用户表 - 10万用户
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INT PRIMARY KEY AUTO_INCREMENT,
                username VARCHAR(50) UNIQUE NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                phone VARCHAR(20),
                first_name VARCHAR(50),
                last_name VARCHAR(50),
                age INT,
                gender ENUM('M', 'F', 'O'),
                city VARCHAR(50),
                country VARCHAR(50),
                registration_date DATE,
                last_login DATETIME,
                loyalty_level ENUM('Bronze', 'Silver', 'Gold', 'Platinum'),
                total_orders INT DEFAULT 0,
                total_spent DECIMAL(12,2) DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_city (city),
                INDEX idx_country (country),
                INDEX idx_registration (registration_date),
                INDEX idx_loyalty (loyalty_level)
            )
        """)
        
        # 产品表 - 5万产品
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INT PRIMARY KEY AUTO_INCREMENT,
                product_name VARCHAR(200) NOT NULL,
                description TEXT,
                category_id INT,
                subcategory_id INT,
                brand VARCHAR(100),
                price DECIMAL(10,2) NOT NULL,
                cost_price DECIMAL(10,2),
                stock_quantity INT DEFAULT 0,
                min_stock_level INT DEFAULT 10,
                weight DECIMAL(8,2),
                dimensions VARCHAR(50),
                color VARCHAR(30),
                size VARCHAR(20),
                is_active BOOLEAN DEFAULT TRUE,
                rating DECIMAL(3,2) DEFAULT 0,
                review_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_category (category_id),
                INDEX idx_brand (brand),
                INDEX idx_price (price),
                INDEX idx_active (is_active)
            )
        """)
        
        # 分类表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                category_id INT PRIMARY KEY AUTO_INCREMENT,
                category_name VARCHAR(100) NOT NULL,
                parent_category_id INT,
                description TEXT,
                level INT DEFAULT 1,
                product_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_parent (parent_category_id)
            )
        """)
        
        # 订单表 - 50万订单
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                order_date DATETIME NOT NULL,
                status ENUM('pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'refunded'),
                total_amount DECIMAL(12,2) NOT NULL,
                shipping_cost DECIMAL(8,2) DEFAULT 0,
                tax_amount DECIMAL(8,2) DEFAULT 0,
                discount_amount DECIMAL(8,2) DEFAULT 0,
                payment_method ENUM('credit_card', 'debit_card', 'paypal', 'bank_transfer', 'cash'),
                payment_status ENUM('pending', 'paid', 'failed', 'refunded'),
                shipping_address TEXT,
                billing_address TEXT,
                tracking_number VARCHAR(100),
                estimated_delivery DATE,
                actual_delivery DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                INDEX idx_user (user_id),
                INDEX idx_order_date (order_date),
                INDEX idx_status (status),
                INDEX idx_payment_status (payment_status)
            )
        """)
        
        # 订单详情表 - 200万订单项
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS order_items (
                order_item_id INT PRIMARY KEY AUTO_INCREMENT,
                order_id INT NOT NULL,
                product_id INT NOT NULL,
                quantity INT NOT NULL,
                unit_price DECIMAL(10,2) NOT NULL,
                discount DECIMAL(8,2) DEFAULT 0,
                line_total DECIMAL(10,2) AS (quantity * (unit_price - discount)),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_order (order_id),
                INDEX idx_product (product_id)
            )
        """)
        
        # 库存表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS inventory (
                inventory_id INT PRIMARY KEY AUTO_INCREMENT,
                product_id INT NOT NULL,
                warehouse_id INT,
                quantity INT NOT NULL,
                reserved_quantity INT DEFAULT 0,
                available_quantity INT AS (quantity - reserved_quantity),
                last_restock_date DATE,
                next_restock_date DATE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_product (product_id),
                INDEX idx_warehouse (warehouse_id)
            )
        """)
        
        # 评论表
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                review_id INT PRIMARY KEY AUTO_INCREMENT,
                user_id INT NOT NULL,
                product_id INT NOT NULL,
                order_id INT,
                rating INT NOT NULL CHECK (rating BETWEEN 1 AND 5),
                title VARCHAR(200),
                comment TEXT,
                is_verified_purchase BOOLEAN DEFAULT FALSE,
                helpful_votes INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                INDEX idx_user (user_id),
                INDEX idx_product (product_id),
                INDEX idx_rating (rating)
            )
        """)
        
        print("✅ 表结构创建完成")
    
    def generate_categories(self):
        """生成产品分类"""
        print("📂 生成产品分类...")
        
        categories = [
            # 一级分类
            ('Electronics', None, 'Electronic devices and accessories'),
            ('Clothing', None, 'Fashion and apparel'),
            ('Home & Garden', None, 'Home improvement and garden supplies'),
            ('Sports & Outdoors', None, 'Sports equipment and outdoor gear'),
            ('Books & Media', None, 'Books, movies, and music'),
            
            # 电子产品子分类
            ('Smartphones', 1, 'Mobile phones and smartphones'),
            ('Laptops', 1, 'Laptops and notebooks'),
            ('Tablets', 1, 'Tablets and e-readers'),
            ('Headphones', 1, 'Audio headphones and earphones'),
            ('Cameras', 1, 'Digital cameras and accessories'),
            
            # 服装子分类
            ('Men Clothing', 2, "Men's fashion and apparel"),
            ('Women Clothing', 2, "Women's fashion and apparel"),
            ('Kids Clothing', 2, "Children's clothing"),
            ('Shoes', 2, 'Footwear for all ages'),
            ('Accessories', 2, 'Fashion accessories'),
        ]
        
        for category in categories:
            self.cursor.execute(
                "INSERT INTO categories (category_name, parent_category_id, description) VALUES (%s, %s, %s)",
                category
            )
        
        self.conn.commit()
        print("✅ 分类数据生成完成")
    
    def generate_users(self, count=100000):
        """生成用户数据"""
        print(f"👥 生成 {count} 个用户数据...")
        
        cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
                 'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'London', 'Paris', 'Tokyo',
                 'Beijing', 'Shanghai', 'Mumbai', 'Sydney', 'Berlin', 'Toronto', 'Singapore']
        
        countries = ['USA', 'UK', 'Canada', 'Australia', 'Germany', 'France', 'Japan', 'China', 'India']
        
        loyalty_levels = ['Bronze', 'Silver', 'Gold', 'Platinum']
        
        batch_size = 1000
        for batch in range(0, count, batch_size):
            current_batch = min(batch_size, count - batch)
            user_data = []
            
            for i in range(current_batch):
                user_id = batch + i + 1
                username = f"user{user_id}"
                email = f"user{user_id}@example.com"
                first_name = random.choice(['John', 'Jane', 'Mike', 'Sarah', 'David', 'Lisa', 'Robert', 'Emily'])
                last_name = random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis'])
                age = random.randint(18, 80)
                gender = random.choice(['M', 'F'])
                city = random.choice(cities)
                country = random.choice(countries)
                registration_date = datetime.now() - timedelta(days=random.randint(1, 3650))
                last_login = registration_date + timedelta(days=random.randint(1, 365))
                loyalty = random.choices(loyalty_levels, weights=[0.4, 0.3, 0.2, 0.1])[0]
                total_orders = random.randint(0, 200)
                total_spent = round(random.uniform(0, 10000), 2)
                
                user_data.append((
                    username, email, f"+1-555-{random.randint(100,999)}-{random.randint(1000,9999)}",
                    first_name, last_name, age, gender, city, country, registration_date,
                    last_login, loyalty, total_orders, total_spent
                ))
            
            # 批量插入
            insert_sql = """
                INSERT INTO users (username, email, phone, first_name, last_name, age, gender, 
                city, country, registration_date, last_login, loyalty_level, total_orders, total_spent)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.executemany(insert_sql, user_data)
            self.conn.commit()
            
            if (batch // batch_size) % 10 == 0:
                print(f"  已生成 {batch + current_batch} 个用户")
        
        print("✅ 用户数据生成完成")
    
    def generate_products(self, count=50000):
        """生成产品数据"""
        print(f"📦 生成 {count} 个产品数据...")
        
        brands = {
            'Electronics': ['Apple', 'Samsung', 'Sony', 'LG', 'Dell', 'HP', 'Canon', 'Nikon'],
            'Clothing': ['Nike', 'Adidas', 'Zara', 'H&M', 'Uniqlo', 'Levi\'s', 'Gucci', 'Prada'],
            'Home & Garden': ['IKEA', 'Home Depot', 'Black & Decker', 'Scotts', 'Weber'],
            'Sports & Outdoors': ['Nike', 'Adidas', 'Under Armour', 'Wilson', 'Spalding'],
            'Books & Media': ['Penguin', 'HarperCollins', 'Random House', 'Disney', 'Warner Bros']
        }
        
        batch_size = 1000
        for batch in range(0, count, batch_size):
            current_batch = min(batch_size, count - batch)
            product_data = []
            
            for i in range(current_batch):
                product_id = batch + i + 1
                category_id = random.randint(1, 15)
                main_category = (category_id - 1) // 3 + 1
                
                if main_category == 1:  # Electronics
                    product_name = f"{random.choice(brands['Electronics'])} {random.choice(['Smartphone', 'Laptop', 'Tablet', 'Headphones', 'Camera'])} {product_id}"
                    price = round(random.uniform(50, 2000), 2)
                elif main_category == 2:  # Clothing
                    product_name = f"{random.choice(brands['Clothing'])} {random.choice(['T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes'])} {product_id}"
                    price = round(random.uniform(10, 300), 2)
                else:  # Other categories
                    product_name = f"Product {product_id}"
                    price = round(random.uniform(5, 500), 2)
                
                description = f"High-quality {product_name} with excellent features"
                brand = random.choice(brands.get(list(brands.keys())[main_category-1], ['Generic']))
                cost_price = round(price * random.uniform(0.3, 0.7), 2)
                stock_quantity = random.randint(0, 1000)
                weight = round(random.uniform(0.1, 20), 2)
                rating = round(random.uniform(3.0, 5.0), 2)
                review_count = random.randint(0, 500)
                
                product_data.append((
                    product_name, description, category_id, random.randint(1, 5),
                    brand, price, cost_price, stock_quantity, 10, weight,
                    f"{random.randint(1,50)}x{random.randint(1,50)}x{random.randint(1,50)}cm",
                    random.choice(['Black', 'White', 'Red', 'Blue', 'Silver']),
                    random.choice(['S', 'M', 'L', 'XL', 'XXL']), True, rating, review_count
                ))
            
            # 批量插入
            insert_sql = """
                INSERT INTO products (product_name, description, category_id, subcategory_id, 
                brand, price, cost_price, stock_quantity, min_stock_level, weight, dimensions, 
                color, size, is_active, rating, review_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.executemany(insert_sql, product_data)
            self.conn.commit()
            
            if (batch // batch_size) % 5 == 0:
                print(f"  已生成 {batch + current_batch} 个产品")
        
        print("✅ 产品数据生成完成")
    
    def generate_orders(self, count=500000):
        """生成订单数据"""
        print(f"🛒 生成 {count} 个订单数据...")
        
        status_weights = [0.1, 0.2, 0.3, 0.3, 0.05, 0.05]  # pending, confirmed, shipped, delivered, cancelled, refunded
        payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer', 'cash']
        payment_status_weights = [0.05, 0.9, 0.03, 0.02]  # pending, paid, failed, refunded
        
        batch_size = 5000
        for batch in range(0, count, batch_size):
            current_batch = min(batch_size, count - batch)
            order_data = []
            
            for i in range(current_batch):
                order_id = batch + i + 1
                user_id = random.randint(1, 100000)
                order_date = datetime.now() - timedelta(days=random.randint(1, 365))
                status = random.choices(['pending', 'confirmed', 'shipped', 'delivered', 'cancelled', 'refunded'], 
                                      weights=status_weights)[0]
                total_amount = round(random.uniform(10, 2000), 2)
                shipping_cost = round(random.uniform(0, 50), 2)
                tax_amount = round(total_amount * 0.08, 2)
                discount_amount = round(total_amount * random.uniform(0, 0.3), 2)
                payment_method = random.choice(payment_methods)
                payment_status = random.choices(['pending', 'paid', 'failed', 'refunded'], 
                                              weights=payment_status_weights)[0]
                
                order_data.append((
                    user_id, order_date, status, total_amount, shipping_cost, tax_amount,
                    discount_amount, payment_method, payment_status,
                    f"{random.randint(100,999)} Main St, City{random.randint(1,100)}",
                    f"{random.randint(100,999)} Main St, City{random.randint(1,100)}",
                    f"TRK{order_id:08d}",
                    order_date + timedelta(days=random.randint(3, 14)),
                    order_date + timedelta(days=random.randint(5, 21)) if status in ['delivered', 'shipped'] else None
                ))
            
            # 批量插入
            insert_sql = """
                INSERT INTO orders (user_id, order_date, status, total_amount, shipping_cost, 
                tax_amount, discount_amount, payment_method, payment_status, shipping_address, 
                billing_address, tracking_number, estimated_delivery, actual_delivery)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.cursor.executemany(insert_sql, order_data)
            self.conn.commit()
            
            if (batch // batch_size) % 10 == 0:
                print(f"  已生成 {batch + current_batch} 个订单")
        
        print("✅ 订单数据生成完成")
    
    def generate_order_items(self, count=2000000):
        """生成订单项数据"""
        print(f"📋 生成 {count} 个订单项数据...")
        
        batch_size = 10000
        for batch in range(0, count, batch_size):
            current_batch = min(batch_size, count - batch)
            order_item_data = []
            
            for i in range(current_batch):
                order_item_id = batch + i + 1
                order_id = random.randint(1, 500000)
                product_id = random.randint(1, 50000)
                quantity = random.randint(1, 5)
                unit_price = round(random.uniform(5, 500), 2)
                discount = round(unit_price * random.uniform(0, 0.2), 2)
                
                order_item_data.append((
                    order_id, product_id, quantity, unit_price, discount
                ))
            
            # 批量插入
            insert_sql = """
                INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
                VALUES (%s, %s, %s, %s, %s)
            """
            self.cursor.executemany(insert_sql, order_item_data)
            self.conn.commit()
            
            if (batch // batch_size) % 20 == 0:
                print(f"  已生成 {batch + current_batch} 个订单项")
        
        print("✅ 订单项数据生成完成")
    
    def create_all_data(self):
        """创建所有数据"""
        start_time = time.time()
        
        self.connect()
        self.create_tables()
        self.generate_categories()
        self.generate_users(100000)      # 10万用户
        self.generate_products(50000)    # 5万产品
        self.generate_orders(500000)     # 50万订单
        self.generate_order_items(2000000)  # 200万订单项
        
        end_time = time.time()
        print(f"\n🎉 所有数据生成完成！耗时: {end_time - start_time:.2f} 秒")
        
        # 显示统计信息
        self.show_statistics()
    
    def show_statistics(self):
        """显示数据库统计信息"""
        print("\n" + "="*50)
        print("📊 数据库统计信息")
        print("="*50)
        
        tables = ['users', 'products', 'orders', 'order_items', 'categories', 'reviews', 'inventory']
        
        for table in tables:
            try:
                self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = self.cursor.fetchone()[0]
                print(f"{table:15} : {count:>10,} 行")
            except:
                print(f"{table:15} : {'表不存在':>10}")
        
        # 显示关系统计
        print("\n🔗 数据关系统计:")
        self.cursor.execute("""
            SELECT '平均订单金额' as metric, ROUND(AVG(total_amount), 2) as value FROM orders
            UNION ALL
            SELECT '总用户数', COUNT(*) FROM users  
            UNION ALL
            SELECT '活跃产品数', COUNT(*) FROM products WHERE is_active = TRUE
            UNION ALL
            SELECT '平均订单项数', ROUND(AVG(item_count), 2) FROM (
                SELECT order_id, COUNT(*) as item_count FROM order_items GROUP BY order_id
            ) as order_counts
        """)
        
        for metric, value in self.cursor.fetchall():
            print(f"  {metric:20} : {value:>15}")

def main():
    db_config = {
        'host': 'localhost',
        'user': 'root',
        'password': 'admin',
        'database': 'test_rag_large'  # 新数据库
    }
    
    # 首先创建数据库
    try:
        conn = mysql.connector.connect(
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password']
        )
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_config['database']}")
        cursor.close()
        conn.close()
        print(f"✅ 数据库 {db_config['database']} 创建成功")
    except Exception as e:
        print(f"❌ 数据库创建失败: {e}")
        return
    
    # 创建大规模数据
    creator = LargeDatabaseCreator(db_config)
    creator.create_all_data()

if __name__ == "__main__":
    main()