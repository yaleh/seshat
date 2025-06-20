#!/usr/bin/env python3
import subprocess
import sys

# Run our specific test files with coverage
test_files = [
    'test_batch_ui',
    'test_chatbot', 
    'test_config_loader',
    'test_dataframe_io',
    'test_db_sqlite3',
    'test_lcel',
    'test_seshat',
    'test_system_ui',
    'test_utils',
    'test_embedding_ui',
    'test_table_parser_extended'
]

# Run coverage on our tests only
cmd = ['coverage', 'run', '--source=.', '--omit=test/*,*/__pycache__/*,*/site-packages/*', '-m', 'unittest', 'discover', '-s', 'test', '-p', 'test_*.py']
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode != 0:
    print("Tests failed:")
    print(result.stderr)
    sys.exit(1)

# Generate coverage report
result = subprocess.run(['coverage', 'report'], capture_output=True, text=True)
print(result.stdout)