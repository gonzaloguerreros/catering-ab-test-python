# conftest.py — pytest configuration at repository root.
#
# Prevent pytest from crawling into src/ and mistaking public functions
# named test_* (e.g. test_conversion_rate) for test cases.
# Reference: pytest docs §"Conftest.py: local per-directory plugins"
collect_ignore_glob = ["src/*.py"]
