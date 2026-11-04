# Run the opt-in data-layer tests against the configured PostgreSQL.
$env:DB_TEST = "1"
.\venv\Scripts\python.exe -m unittest tests.test_db_integration -v
