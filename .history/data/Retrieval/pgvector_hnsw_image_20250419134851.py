import psycopg2

# --- PostgreSQL Config ---
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "Tuti1311"
HOST = "localhost"
PORT = 5433

# --- Create HNSW index on keyframes ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT
)
cur = conn.cursor()

cur.execute("""
DROP INDEX IF EXISTS keyframe_embedding_hnsw;
CREATE INDEX keyframe_embedding_hnsw
ON keyframes
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
""")

conn.commit()
cur.close()
conn.close()
print("✅ HNSW index created for keyframes.")
