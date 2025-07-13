# # db_utils.py
# import psycopg2

# def get_db_connection():
#     try:
#         conn = psycopg2.connect(
#             host="localhost", # Your database host
#             database="your_database_name", # Your database name
#             user="your_username", # Your database username
#             password="your_password" # Your database password
#         )
#         return conn
#     except psycopg2.Error as e:
#         print(f"Error connecting to the database: {e}")
#         raise # Re-raise the exception to be caught in app.py