import sqlite3

def list_tables(cursor):
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    return [table[0] for table in tables]

def list_columns(cursor, table_name):
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = cursor.fetchall()
    return [column[1] for column in columns]

def fetch_all_data(cursor, table_name):
    cursor.execute(f"SELECT * FROM {table_name};")
    rows = cursor.fetchall()
    return rows

def main(database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    tables = list_tables(cursor)
    print(f"Tables in the database: {tables}")

    for table in tables:
        print(f"\nTable: {table}")
        columns = list_columns(cursor, table)
        print(f"Columns: {columns}")

        rows = fetch_all_data(cursor, table)
        for row in rows:
            print(row)

    conn.close()

if __name__ == "__main__":
    database_path = "langchain_cache.db"  # 替换为你的数据库文件路径
    main(database_path)