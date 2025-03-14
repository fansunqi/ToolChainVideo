import sqlite3

# 连接 SQLite 数据库
conn = sqlite3.connect("langchain_cache.db")
cursor = conn.cursor()

# 查看表结构
cursor.execute("PRAGMA table_info(langchain_sqlite_cache);")
columns = cursor.fetchall()
print("表结构:", columns)

# 查询所有缓存数据
cursor.execute("SELECT * FROM langchain_sqlite_cache;")
rows = cursor.fetchall()

# 打印缓存内容
for row in rows:
    print(row)

# 关闭数据库连接
conn.close()
