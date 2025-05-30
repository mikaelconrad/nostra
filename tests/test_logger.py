"""
Tests for the logger module
"""

import pytest
import os
import logging
from utils.logger import setup_logger, ColoredFormatter, log_exception

class TestLogger:
    """Test suite for logger functionality"""
    
    def test_setup_logger(self):
        """Test logger setup"""
        logger = setup_logger('test_logger')
        
        assert logger is not None
        assert logger.name == 'test_logger'
        assert logger.level == logging.INFO
        
        # Check handlers
        assert len(logger.handlers) >= 2  # File and console handlers
        
        # Check if file handler exists
        file_handlers = [h for h in logger.handlers if hasattr(h, 'baseFilename')]
        assert len(file_handlers) > 0
    
    def test_logger_singleton(self):
        """Test that multiple calls return the same logger instance"""
        logger1 = setup_logger('test_singleton')
        logger2 = setup_logger('test_singleton')
        
        assert logger1 is logger2
        # Should not add duplicate handlers
        assert len(logger1.handlers) == len(logger2.handlers)
    
    def test_colored_formatter(self):
        """Test colored formatter"""
        formatter = ColoredFormatter('%(levelname)s - %(message)s')
        
        # Create log records
        record = logging.LogRecord(
            name='test',
            level=logging.INFO,
            pathname='test.py',
            lineno=1,
            msg='Test message',
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain ANSI color codes
        assert '\x1b[' in formatted
        assert 'Test message' in formatted
    
    def test_log_exception_decorator(self):
        """Test log exception decorator"""
        logger = setup_logger('test_decorator')
        
        @log_exception(logger)
        def failing_function():
            raise ValueError("Test error")
        
        # Function should raise the exception
        with pytest.raises(ValueError) as exc:
            failing_function()
        
        assert str(exc.value) == "Test error"
    
    def test_log_levels(self):
        """Test different log levels"""
        logger = setup_logger('test_levels')
        
        # These should not raise exceptions
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")
        
        # Test with exception
        try:
            raise Exception("Test exception")
        except Exception:
            logger.exception("Exception occurred")
    
    def test_log_file_creation(self, temp_data_dir):
        """Test that log files are created"""
        import config
        log_dir = os.path.join(temp_data_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Temporarily override log directory
        original_log_dir = getattr(config, 'LOG_DIRECTORY', None)
        config.LOG_DIRECTORY = log_dir
        
        logger = setup_logger('test_file_creation')
        logger.info("Test log message")
        
        # Check if log file was created
        log_files = os.listdir(log_dir)
        assert any('test_file_creation.log' in f for f in log_files)
        
        # Restore original
        if original_log_dir:
            config.LOG_DIRECTORY = original_log_dir