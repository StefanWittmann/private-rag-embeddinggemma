# Database setup code to replace in your Jupyter notebook cell:
# This replaces the problematic SQL creation code

# Initialize SQLite with vector extension
print("🗄️ Setting up vector database...")
conn = sqlite3.connect(DB_FILE)
conn.enable_load_extension(True)
sqlite_vec.load(conn)
conn.enable_load_extension(False)

# Create vector table - FIXED VERSION (single line f-string)
create_table_sql = f"CREATE VIRTUAL TABLE IF NOT EXISTS {TABLE_NAME} USING vec0(text TEXT, source TEXT, embedding float[{EMBEDDING_DIMS}])"
conn.execute(create_table_sql)
conn.commit()
print("✅ Vector database ready!")