import sqlite3

# 替换为你的 SQLite 数据库文件路径
db_path = "/home/fsq/video_agent/ToolChainVideo/langchain_cache.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("数据库中的表:", tables)

for table in tables:
    table_name = table[0]
    print(f"\n表 {table_name} 的结构:")
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    for col in columns:
        print(col)  # (列ID, 列名, 数据类型, 是否允许NULL, 默认值, 主键标志)

for table in tables:
    table_name = table[0]
    print(f"\n表 {table_name} 的前5条数据:")
    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
    rows = cursor.fetchall()
    for row in rows:
        print(row)

