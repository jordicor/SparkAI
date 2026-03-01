import sqlite3

# Connect to Prompt.db database
prompt_conn = sqlite3.connect('Prompt.db')
prompt_cursor = prompt_conn.cursor()

# Connect to Aurvek.db database
aurvek_conn = sqlite3.connect('Aurvek.db')
aurvek_cursor = aurvek_conn.cursor()

# Delete existing data in Aurvek.db tables
aurvek_cursor.execute("DELETE FROM SERVICES")
aurvek_cursor.execute("DELETE FROM VOICES")
aurvek_cursor.execute("DELETE FROM PROMPTS")
aurvek_cursor.execute("DELETE FROM LLM")  # Add this line to clear the LLM table

# Extract data from SERVICES table from Prompt.db
prompt_cursor.execute("SELECT id, name, unit, cost_per_unit, type FROM SERVICES")
services_data = prompt_cursor.fetchall()

# Insert data into the SERVICES table in Aurvek.db
aurvek_cursor.executemany("INSERT INTO SERVICES (id, name, unit, cost_per_unit, type) VALUES (?, ?, ?, ?, ?)", services_data)

# Extract data from VOICES table from Prompt.db
prompt_cursor.execute("SELECT id, name, voice_code, tts_service FROM VOICES")
voices_data = prompt_cursor.fetchall()

# Insert data into the VOICES table in Aurvek.db
aurvek_cursor.executemany("INSERT INTO VOICES (id, name, voice_code, tts_service) VALUES (?, ?, ?, ?)", voices_data)

# Extract data from PROMPTS table from Prompt.db
prompt_cursor.execute("SELECT id, name, prompt, voice_id FROM PROMPTS")
prompts_data = prompt_cursor.fetchall()

# Insert data into the PROMPTS table in Aurvek.db
aurvek_cursor.executemany("INSERT INTO PROMPTS (id, name, prompt, voice_id) VALUES (?, ?, ?, ?)", prompts_data)

# Extract data from LLM table from Prompt.db
prompt_cursor.execute("SELECT id, machine, model, input_token_cost, output_token_cost, vision FROM LLM")
llm_data = prompt_cursor.fetchall()

# Insert data into the LLM table in Aurvek.db
aurvek_cursor.executemany("INSERT INTO LLM (id, machine, model, input_token_cost, output_token_cost, vision) VALUES (?, ?, ?, ?, ?, ?)", llm_data)

# Confirm changes and close connections
aurvek_conn.commit()
prompt_conn.close()
aurvek_conn.close()

print("Data successfully transferred from Prompt.db to Aurvek.db, including the LLM table")