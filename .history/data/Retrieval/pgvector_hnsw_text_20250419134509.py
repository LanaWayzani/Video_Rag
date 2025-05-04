import psycopg2

# --- PostgreSQL Config ---
DB_NAME = "videoqa"
USER = "postgres"
PASSWORD = "your_password"
HOST = "localhost"
PORT = 5433

# --- Create HNSW index on transcript_segments ---
conn = psycopg2.connect(
    dbname=DB_NAME, user=USER, password=PASSWORD, host=HOST, port=PORT
)
cur = conn.cursor()

cur.execute("""
DROP INDEX IF EXISTS transcript_embedding_hnsw;
CREATE INDEX transcript_embedding_hnsw
ON transcript_segments
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
""")

conn.commit()
cur.close()
conn.close()
print("✅ HNSW index created for transcript_segments.")
