import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

# Parse DATABASE_URL: mysql+pymysql://root:root321@localhost/air_quality_db
db_url = os.environ.get('DATABASE_URL')
# Simple parsing
# Remove mysql+pymysql://
clean_url = db_url.replace('mysql+pymysql://', '')
user_pass, host_db = clean_url.split('@')
user, password = user_pass.split(':')
host, db_name = host_db.split('/')

try:
    connection = pymysql.connect(
        host=host,
        user=user,
        password=password,
        database=db_name
    )
    with connection.cursor() as cursor:
        # Check if role column already exists
        cursor.execute("SHOW COLUMNS FROM users LIKE 'role'")
        result = cursor.fetchone()
        if not result:
            sql = "ALTER TABLE users ADD COLUMN role VARCHAR(20) DEFAULT 'user'"
            cursor.execute(sql)
            connection.commit()
            print("Successfully added 'role' column to users table.")
        else:
            print("'role' column already exists.")
except Exception as e:
    print(f"Error updating database: {e}")
finally:
    if 'connection' in locals():
        connection.close()
