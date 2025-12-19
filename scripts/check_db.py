"""Script to test database connection and pgvector extension."""

import asyncio
import sys
import os

# Add the project root to sys.path to allow importing from 'app'
sys.path.append(os.getcwd())

from app.database import db

async def test_db_connection():
    try:
        print("🔍 Attempting to connect to the database...")
        await db.connect()
        
        # Check if we can run a query and check for extension
        async with db.pool.acquire() as conn:
            # Check for pgvector
            version = await conn.fetchval("SELECT version()")
            print(f"✅ Connected to: {version}")
            
            # Check for vector extension
            ext_exists = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )
            if ext_exists:
                print("✅ pgvector extension is INSTALLED and ENABLED.")
                
                # Test a simple vector operation
                await conn.execute("CREATE TEMP TABLE test_vector (v vector(3))")
                await conn.execute("INSERT INTO test_vector (v) VALUES ('[1,2,3]')")
                vec = await conn.fetchval("SELECT v FROM test_vector")
                print(f"✅ Vector operation successful. Read vector: {vec}")
            else:
                print("❌ pgvector extension NOT FOUND.")
                
        await db.disconnect()
        print("✨ Database check complete!")
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print("\nPossible issues:")
        print("1. Is the Docker container running? (Run 'docker ps')")
        print("2. Is the DATABASE_URL in .env correct?")
        print("3. Did you use the correct password (postgres)?")

if __name__ == "__main__":
    asyncio.run(test_db_connection())
