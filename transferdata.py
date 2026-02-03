import sqlite3

# Connect to Prompt.db database
prompt_conn = sqlite3.connect('Prompt.db')
prompt_cursor = prompt_conn.cursor()

# Connect to Spark.db database
spark_conn = sqlite3.connect('Spark.db')
spark_cursor = spark_conn.cursor()

# Delete existing data in Spark.db tables
spark_cursor.execute("DELETE FROM SERVICES")
spark_cursor.execute("DELETE FROM VOICES")
spark_cursor.execute("DELETE FROM PROMPTS")
spark_cursor.execute("DELETE FROM LLM")  # Add this line to clear the LLM table

# Extract data from SERVICES table from Prompt.db
prompt_cursor.execute("SELECT id, name, unit, cost_per_unit, type FROM SERVICES")
services_data = prompt_cursor.fetchall()

# Insert data into the SERVICES table in Spark.db
spark_cursor.executemany("INSERT INTO SERVICES (id, name, unit, cost_per_unit, type) VALUES (?, ?, ?, ?, ?)", services_data)

# Extract data from VOICES table from Prompt.db
prompt_cursor.execute("SELECT id, name, voice_code, tts_service FROM VOICES")
voices_data = prompt_cursor.fetchall()

# Insert data into the VOICES table in Spark.db
spark_cursor.executemany("INSERT INTO VOICES (id, name, voice_code, tts_service) VALUES (?, ?, ?, ?)", voices_data)

# Extract data from PROMPTS table from Prompt.db
prompt_cursor.execute("SELECT id, name, prompt, voice_id FROM PROMPTS")
prompts_data = prompt_cursor.fetchall()

# Insert data into the PROMPTS table in Spark.db
spark_cursor.executemany("INSERT INTO PROMPTS (id, name, prompt, voice_id) VALUES (?, ?, ?, ?)", prompts_data)

# Extract data from LLM table from Prompt.db
prompt_cursor.execute("SELECT id, machine, model, input_token_cost, output_token_cost, vision FROM LLM")
llm_data = prompt_cursor.fetchall()

# Insert data into the LLM table in Spark.db
spark_cursor.executemany("INSERT INTO LLM (id, machine, model, input_token_cost, output_token_cost, vision) VALUES (?, ?, ?, ?, ?, ?)", llm_data)

# Confirm changes and close connections
spark_conn.commit()
prompt_conn.close()
spark_conn.close()

print("Data successfully transferred from Prompt.db to Spark.db, including the LLM table")