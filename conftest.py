def pytest_configure(config):
        config.addinivalue_line("markers", "integration: This is an integration test")