import sqlite3

# 数据库文件路径
db_path = "/home/fsq/.cache/octotools/cache_openai_gpt-4o.db/cache.db"

# 连接到 SQLite 数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询表的行数
table_name = "Cache"  # 替换为你的表名
cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
row_count = cursor.fetchone()[0]

print(f"表 {table_name} 中的行数: {row_count}")

# 关闭连接
cursor.close()
conn.close()