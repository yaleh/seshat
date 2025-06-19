import unittest
import tempfile
import os
import sqlite3
from unittest.mock import patch, MagicMock
import datetime

from db.db_sqlite3 import (
    DatabaseManager, USER_MESSAGES_TABLE, SYSTEM_MESSAGES_TABLE,
    STRING_MESSAGES_TABLE, LANGSERVE_MESSAGES_TABLE, LANGSERVE_URLS_TABLE,
    EMBEDDING_PINECONE_HOSTS_TABLE, EMBEDDING_PINECONE_API_KEYS_TABLE,
    EMBEDDING_MILVUS_URIS_TABLE, EMBEDDING_MILVUS_TOKENS_TABLE,
    EMBEDDING_MILVUS_COLLECTIONS_TABLE, tables
)


class TestDatabaseManager(unittest.TestCase):
    
    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        self.db_manager = DatabaseManager(self.db_path, max_message_length=1000, max_message_count=5)
    
    def tearDown(self):
        """Clean up test database"""
        self.db_manager.conn.close()
        os.unlink(self.db_path)
    
    def test_init_creates_tables(self):
        """Test that initialization creates all required tables"""
        cursor = self.db_manager.conn.cursor()
        
        # Check that all tables exist
        for table in tables:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,))
            result = cursor.fetchone()
            self.assertIsNotNone(result, f"Table {table} was not created")
            self.assertEqual(result[0], table)
    
    def test_init_sets_properties(self):
        """Test that initialization sets max_message_length and max_message_count"""
        self.assertEqual(self.db_manager.max_message_length, 1000)
        self.assertEqual(self.db_manager.max_message_count, 5)
    
    def test_create_table(self):
        """Test creating a custom table"""
        table_name = "test_table"
        columns = ["id INTEGER PRIMARY KEY", "name TEXT", "value INTEGER"]
        
        self.db_manager.create_table(table_name, columns)
        
        # Verify table exists
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        result = cursor.fetchone()
        self.assertIsNotNone(result)
        
        # Verify table structure
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns_info = cursor.fetchall()
        self.assertEqual(len(columns_info), 3)
    
    def test_append_message_new(self):
        """Test appending a new message"""
        table_name = USER_MESSAGES_TABLE
        message = "Test message"
        
        with patch('db.db_sqlite3.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1, 12, 0, 0)
            
            self.db_manager.append_message(table_name, message)
        
        # Verify message was inserted
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        result = cursor.fetchall()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], message)
    
    def test_append_message_existing(self):
        """Test appending an existing message updates timestamp"""
        table_name = USER_MESSAGES_TABLE
        message = "Test message"
        
        # Add message first time
        with patch('db.db_sqlite3.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1, 12, 0, 0)
            self.db_manager.append_message(table_name, message)
        
        # Add same message again with different timestamp
        with patch('db.db_sqlite3.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 2, 12, 0, 0)
            self.db_manager.append_message(table_name, message)
        
        # Verify only one message exists with updated timestamp
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT * FROM {table_name}")
        result = cursor.fetchall()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][1], message)
        self.assertEqual(result[0][0], "2024-01-02 12:00:00")
    
    def test_append_message_max_count_exceeded(self):
        """Test that oldest message is deleted when max count is exceeded"""
        table_name = USER_MESSAGES_TABLE
        
        # Add messages up to max count + 1
        for i in range(6):  # max_message_count is 5
            with patch('db.db_sqlite3.datetime') as mock_datetime:
                mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1, 12, i, 0)
                self.db_manager.append_message(table_name, f"Message {i}")
        
        # Verify only max_message_count messages exist
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        self.assertEqual(count, 5)
        
        # Verify oldest message was deleted
        cursor.execute(f"SELECT message FROM {table_name} ORDER BY timestamp")
        messages = [row[0] for row in cursor.fetchall()]
        self.assertNotIn("Message 0", messages)
        self.assertIn("Message 5", messages)
    
    def test_message_exists_true(self):
        """Test message_exists returns True for existing message"""
        table_name = USER_MESSAGES_TABLE
        message = "Test message"
        
        # Add message
        self.db_manager.append_message(table_name, message)
        
        # Check if it exists
        exists = self.db_manager.message_exists(table_name, message)
        self.assertTrue(exists)
    
    def test_message_exists_false(self):
        """Test message_exists returns False for non-existing message"""
        table_name = USER_MESSAGES_TABLE
        message = "Non-existing message"
        
        exists = self.db_manager.message_exists(table_name, message)
        self.assertFalse(exists)
    
    def test_get_messages_empty_table(self):
        """Test get_messages returns empty list for empty table"""
        table_name = USER_MESSAGES_TABLE
        
        messages = self.db_manager.get_messages(table_name)
        self.assertEqual(messages, [])
    
    def test_get_messages_with_data(self):
        """Test get_messages returns messages in descending timestamp order"""
        table_name = USER_MESSAGES_TABLE
        
        # Add multiple messages with different timestamps
        messages_to_add = ["First message", "Second message", "Third message"]
        for i, msg in enumerate(messages_to_add):
            with patch('db.db_sqlite3.datetime') as mock_datetime:
                mock_datetime.datetime.now.return_value = datetime.datetime(2024, 1, 1, 12, i, 0)
                self.db_manager.append_message(table_name, msg)
        
        # Get messages
        retrieved_messages = self.db_manager.get_messages(table_name)
        
        # Should be in reverse order (newest first)
        expected_order = ["Third message", "Second message", "First message"]
        self.assertEqual(retrieved_messages, expected_order)
    
    def test_get_messages_truncates_long_messages(self):
        """Test get_messages truncates messages longer than max_message_length"""
        table_name = USER_MESSAGES_TABLE
        long_message = "x" * 1500  # Longer than max_message_length (1000)
        
        self.db_manager.append_message(table_name, long_message)
        messages = self.db_manager.get_messages(table_name)
        
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]), 1000)  # Truncated to max_message_length
        self.assertEqual(messages[0], "x" * 1000)
    
    def test_get_messages_handles_none(self):
        """Test get_messages handles None message values"""
        table_name = USER_MESSAGES_TABLE
        
        # Manually insert None message
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"INSERT INTO {table_name} VALUES (?, ?)", ("2024-01-01 12:00:00", None))
        self.db_manager.conn.commit()
        
        messages = self.db_manager.get_messages(table_name)
        self.assertEqual(messages, [''])  # None should become empty string
    
    def test_update_message_timestamp(self):
        """Test updating message timestamp"""
        table_name = USER_MESSAGES_TABLE
        message = "Test message"
        
        # Add message first
        self.db_manager.append_message(table_name, message)
        
        # Update timestamp
        with patch('db.db_sqlite3.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = datetime.datetime(2024, 2, 1, 12, 0, 0)
            self.db_manager.update_message_timestamp(table_name, message)
        
        # Verify timestamp was updated
        cursor = self.db_manager.conn.cursor()
        cursor.execute(f"SELECT timestamp FROM {table_name} WHERE message=?", (message,))
        result = cursor.fetchone()
        self.assertEqual(result[0], "2024-02-01 12:00:00")


class TestDatabaseConstants(unittest.TestCase):
    
    def test_table_constants(self):
        """Test that all table constants are defined"""
        expected_tables = [
            "user_messages", "system_messages", "string_messages",
            "langserve_messages", "langserve_urls", "embedding_pinecone_hosts",
            "embedding_pinecone_api_keys", "embedding_milvus_uris",
            "embedding_milvus_tokens", "embedding_milvus_collections"
        ]
        
        # Check individual constants
        self.assertEqual(USER_MESSAGES_TABLE, "user_messages")
        self.assertEqual(SYSTEM_MESSAGES_TABLE, "system_messages")
        self.assertEqual(STRING_MESSAGES_TABLE, "string_messages")
        self.assertEqual(LANGSERVE_MESSAGES_TABLE, "langserve_messages")
        self.assertEqual(LANGSERVE_URLS_TABLE, "langserve_urls")
        self.assertEqual(EMBEDDING_PINECONE_HOSTS_TABLE, "embedding_pinecone_hosts")
        self.assertEqual(EMBEDDING_PINECONE_API_KEYS_TABLE, "embedding_pinecone_api_keys")
        self.assertEqual(EMBEDDING_MILVUS_URIS_TABLE, "embedding_milvus_uris")
        self.assertEqual(EMBEDDING_MILVUS_TOKENS_TABLE, "embedding_milvus_tokens")
        self.assertEqual(EMBEDDING_MILVUS_COLLECTIONS_TABLE, "embedding_milvus_collections")
        
        # Check tables list
        self.assertEqual(len(tables), len(expected_tables))
        for table in expected_tables:
            self.assertIn(table, tables)


if __name__ == '__main__':
    unittest.main()