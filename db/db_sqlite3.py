import datetime
import sqlite3

USER_MESSAGES_TABLE = "user_messages"
SYSTEM_MESSAGES_TABLE = "system_messages"
STRING_MESSAGES_TABLE = "string_messages"
LANGSERVE_MESSAGES_TABLE = "langserve_messages"
LANGSERVE_URLS_TABLE = "langserve_urls"

tables = [
    USER_MESSAGES_TABLE,
    SYSTEM_MESSAGES_TABLE,
    STRING_MESSAGES_TABLE,
    LANGSERVE_MESSAGES_TABLE,
    LANGSERVE_URLS_TABLE
]

class DatabaseManager:
    def __init__(self, database_name, max_message_length=65535):
        self.conn = sqlite3.connect(database_name, check_same_thread=False)

        for table in tables:
            self.create_table(table, ["timestamp TEXT", "message TEXT"])
        
        self.max_message_length = max_message_length

    def create_table(self, table_name, columns):
        c = self.conn.cursor()
        column_str = ", ".join(columns)
        c.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({column_str})")
        self.conn.commit()

    def update_message_timestamp(self, table_name, message):
        c = self.conn.cursor()
        timestamp = str(datetime.datetime.now())
        c.execute(f"UPDATE {table_name} SET timestamp=? WHERE message=?", (str(timestamp), message))
        self.conn.commit()

    def message_exists(self, table_name, message):
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM {table_name} WHERE message=?", (message,))
        return c.fetchone() is not None

    def get_messages(self, table_name):
        c = self.conn.cursor()
        c.execute(f"SELECT * FROM {table_name} ORDER BY timestamp DESC")
        messages = c.fetchall()
        return [message[1][:self.max_message_length] if message[1]!=None else '' for message in messages]

    def append_message(self, table_name, message):
        c = self.conn.cursor()
        timestamp = str(datetime.datetime.now())
        if self.message_exists(table_name, message):
            c.execute(f"UPDATE {table_name} SET timestamp=? WHERE message=?", (timestamp, message))
        else:
            c.execute(f"INSERT INTO {table_name} VALUES (?, ?)", (timestamp, message))
        self.conn.commit()