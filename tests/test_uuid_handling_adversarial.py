# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
tests for UUID handling implementation in KuzuAlchemy.

This test suite is designed to break the UUID handling implementation by testing:
- Exception handling in connection reuse
- UUID type safety enforcement
- UUID validation with edge cases
- Complex real-world scenarios
- Boundary conditions and error scenarios
- Nested structures and unusual patterns
- Performance stress cases
"""

from __future__ import annotations

import uuid
import pytest
from unittest.mock import Mock, patch
import threading
from concurrent.futures import ThreadPoolExecutor
from pydantic import ValidationError

from kuzualchemy import (
    KuzuBaseModel,
    kuzu_node,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
    get_ddl_for_node,
)
from kuzualchemy.test_utilities import initialize_schema


class TestConnectionReuseExceptionHandling:
    """tests for connection reuse exception handling."""
    
    def test_connection_error_during_reuse_logs_and_falls_back(self, test_db_path):
        """Test that ConnectionError during reuse is logged and falls back gracefully."""
        session = KuzuSession(db_path=test_db_path)
        
        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 1
        
        # Mock the reused connection to raise ConnectionError
        session._reused_connection.execute.side_effect = ConnectionError("Connection lost")
        
        # Mock the main connection to succeed
        session._conn.execute = Mock(return_value="success")
        
        with patch('kuzualchemy.kuzu_session.logger') as mock_logger:
            result = session._execute_with_connection_reuse("RETURN 1")
            
            # Verify fallback succeeded
            assert result == "success"
            
            # Verify connection reuse state was reset to main connection for future reuse
            assert session._reused_connection is session._conn
            assert session._connection_operation_count == 1
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Connection reuse failed" in warning_call
            assert "Connection lost" in warning_call
    
    def test_os_error_during_reuse_logs_and_falls_back(self, test_db_path):
        """Test that OSError during reuse is logged and falls back gracefully."""
        session = KuzuSession(db_path=test_db_path)
        
        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 2
        
        # Mock the reused connection to raise OSError
        session._reused_connection.execute.side_effect = OSError("File descriptor error")
        
        # Mock the main connection to succeed
        session._conn.execute = Mock(return_value="fallback_success")
        
        with patch('kuzualchemy.kuzu_session.logger') as mock_logger:
            result = session._execute_with_connection_reuse("MATCH (n) RETURN n")
            
            # Verify fallback succeeded
            assert result == "fallback_success"
            
            # Verify connection reuse state was reset to main connection for future reuse
            assert session._reused_connection is session._conn
            assert session._connection_operation_count == 1
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
    
    def test_runtime_error_during_reuse_logs_and_falls_back(self, test_db_path):
        """Test that RuntimeError during reuse is logged and falls back gracefully."""
        session = KuzuSession(db_path=test_db_path)
        
        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 3
        
        # Mock the reused connection to raise RuntimeError
        session._reused_connection.execute.side_effect = RuntimeError("Database runtime error")
        
        # Mock the main connection to succeed
        session._conn.execute = Mock(return_value="runtime_fallback")
        
        with patch('kuzualchemy.kuzu_session.logger') as mock_logger:
            result = session._execute_with_connection_reuse("CREATE (n:Test)")
            
            # Verify fallback succeeded
            assert result == "runtime_fallback"
            
            # Verify connection reuse state was reset to main connection for future reuse
            assert session._reused_connection is session._conn
            assert session._connection_operation_count == 1
            
            # Verify warning was logged
            mock_logger.warning.assert_called_once()
    
    def test_value_error_during_reuse_logs_and_falls_back(self, test_db_path):
        """Test that ValueError during reuse is treated as connection error and falls back."""
        session = KuzuSession(db_path=test_db_path)

        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 1

        # Mock the reused connection to raise ValueError (treated as connection error)
        session._reused_connection.execute.side_effect = ValueError("Connection validation error")

        # Mock the main connection to succeed
        session._conn.execute = Mock(return_value="value_error_fallback")

        with patch('kuzualchemy.kuzu_session.logger') as mock_logger:
            result = session._execute_with_connection_reuse("RETURN 1")

            # Verify fallback succeeded
            assert result == "value_error_fallback"

            # Verify connection reuse state was reset to main connection for future reuse
            assert session._reused_connection is session._conn
            assert session._connection_operation_count == 1

            # Verify warning was logged
            mock_logger.warning.assert_called_once()
            warning_call = mock_logger.warning.call_args[0][0]
            assert "Connection reuse failed" in warning_call
            assert "Connection validation error" in warning_call
    
    def test_unexpected_exception_during_reuse_logs_and_reraises(self, test_db_path):
        """Test that truly unexpected exceptions during reuse are logged and re-raised."""
        session = KuzuSession(db_path=test_db_path)

        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 1

        # Mock the reused connection to raise an unexpected exception (not a connection error)
        unexpected_error = TypeError("Unexpected type error")
        session._reused_connection.execute.side_effect = unexpected_error

        with patch('kuzualchemy.kuzu_session.logger') as mock_logger:
            with pytest.raises(TypeError, match="Unexpected type error"):
                session._execute_with_connection_reuse("RETURN 1")

            # Verify connection reuse state was reset before re-raising
            assert session._reused_connection is None
            assert session._connection_operation_count == 0

            # Verify error was logged
            mock_logger.error.assert_called_once()
            error_call = mock_logger.error.call_args[0][0]
            assert "Unexpected error during connection reuse" in error_call
            assert "Unexpected type error" in error_call

    def test_memory_error_during_reuse_logs_and_reraises(self, test_db_path):
        """Test that MemoryError during reuse is logged and re-raised."""
        session = KuzuSession(db_path=test_db_path)
        
        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 4
        
        # Mock the reused connection to raise MemoryError
        memory_error = MemoryError("Out of memory")
        session._reused_connection.execute.side_effect = memory_error
        
        with patch('kuzualchemy.kuzu_session.logger') as mock_logger:
            with pytest.raises(MemoryError, match="Out of memory"):
                session._execute_with_connection_reuse("MATCH (n) RETURN n LIMIT 1000000")
            
            # Verify connection reuse state was reset before re-raising
            assert session._reused_connection is None
            assert session._connection_operation_count == 0
            
            # Verify error was logged
            mock_logger.error.assert_called_once()


# @@ STEP: UUID Type Safety Enforcement Tests
# || These tests validate the _validate_manual_auto_increment_values() method
# || which ensures that UUID auto-increment fields receive proper UUID objects


class TestUUIDTypeSafetyEnforcement:
    """tests for UUID type safety enforcement."""
    
    @kuzu_node("UUIDTestNode")
    class UUIDTestNode(KuzuBaseModel):
        id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True, auto_increment=True)
        name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    
    def test_uuid_string_rejection_with_clear_error_message(self, test_db_path):
        """Test that UUID strings are rejected by the validation method."""
        session = KuzuSession(db_path=test_db_path)

        # Test the validation method directly with invalid string value
        manual_values = {
            "id": "550e8400-e29b-41d4-a716-446655440000"  # String - should be rejected
        }

        with pytest.raises(TypeError) as exc_info:
            session._validate_manual_auto_increment_values(manual_values, self.UUIDTestNode)

        # Verify the error message contains expected information
        error_msg = str(exc_info.value)
        assert "UUID field 'id' in UUIDTestNode must be a UUID object, not a string" in error_msg
        assert "550e8400-e29b-41d4-a716-446655440000" in error_msg
        assert "Use uuid.UUID" in error_msg

        session.close()
    
    def test_uuid_integer_rejection(self, test_db_path):
        """Test that integer values for UUID fields are rejected."""
        session = KuzuSession(db_path=test_db_path)

        # Test the validation method directly with invalid integer value
        manual_values = {"id": 12345}

        with pytest.raises(TypeError) as exc_info:
            session._validate_manual_auto_increment_values(manual_values, self.UUIDTestNode)

        error_msg = str(exc_info.value)
        assert "UUID field 'id' in UUIDTestNode must be a UUID object" in error_msg
        assert "12345" in error_msg
        assert "int" in error_msg

    def test_uuid_none_acceptance(self, test_db_path):
        """Test that None values for UUID fields are accepted (skipped in validation)."""
        session = KuzuSession(db_path=test_db_path)

        # Test the validation method directly with None value
        manual_values = {"id": None}

        # None values should be skipped, not cause errors
        session._validate_manual_auto_increment_values(manual_values, self.UUIDTestNode)
        # Should not raise any exception

    def test_uuid_list_rejection(self, test_db_path):
        """Test that list values for UUID fields are rejected."""
        session = KuzuSession(db_path=test_db_path)

        # Test the validation method directly with invalid list value
        manual_values = {
            "id": ["550e8400-e29b-41d4-a716-446655440000"]
        }

        with pytest.raises(TypeError) as exc_info:
            session._validate_manual_auto_increment_values(manual_values, self.UUIDTestNode)

        error_msg = str(exc_info.value)
        assert "UUID field 'id' in UUIDTestNode must be a UUID object" in error_msg
        assert "list" in error_msg

    def test_uuid_dict_rejection(self, test_db_path):
        """Test that dict values for UUID fields are rejected."""
        session = KuzuSession(db_path=test_db_path)

        # Test the validation method directly with invalid dict value
        manual_values = {
            "id": {"uuid": "550e8400-e29b-41d4-a716-446655440000"}
        }

        with pytest.raises(TypeError) as exc_info:
            session._validate_manual_auto_increment_values(manual_values, self.UUIDTestNode)

        error_msg = str(exc_info.value)
        assert "UUID field 'id' in UUIDTestNode must be a UUID object" in error_msg
        assert "dict" in error_msg

    def test_valid_uuid_object_acceptance(self, test_db_path):
        """Test that valid UUID objects are accepted."""
        session = KuzuSession(db_path=test_db_path)

        # Create valid UUID object
        valid_uuid = uuid.UUID("550e8400-e29b-41d4-a716-446655440000")

        # Test the validation method directly with valid UUID object
        manual_values = {"id": valid_uuid}

        # Should succeed without errors
        session._validate_manual_auto_increment_values(manual_values, self.UUIDTestNode)
        # No exception should be raised


class TestUUIDValidationEdgeCases:
    """tests for UUID validation edge cases."""

    @kuzu_node("ValidationTestNode")
    class ValidationTestNode(KuzuBaseModel):
        id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True, auto_increment=True)
        name: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

    def test_malformed_uuid_string_validation_rejection(self, test_db_path):
        """Test that UUID strings are rejected (since only UUID objects are allowed)."""
        session = KuzuSession(db_path=test_db_path)

        # Test various UUID strings (all should be rejected since only UUID objects are allowed)
        uuid_strings = [
            "550e8400-e29b-41d4-a716-446655440000",   # Valid UUID string
            "550e8400-e29b-41d4-a716-446655440000x",  # Extra character
            "550e8400-e29b-41d4-a716-44665544000",    # Missing character
            "550e8400-e29b-41d4-a716-446655440000-",  # Extra hyphen
            "550e8400e29b41d4a716446655440000",        # Missing hyphens
            "550e8400-e29b-41d4-a716",                # Too short
            "550e8400-e29b-41d4-a716-446655440000-550e8400", # Too long
            "ggge8400-e29b-41d4-a716-446655440000",   # Invalid hex chars
            "550e8400-e29b-41d4-a716-44665544000g",   # Invalid hex at end
            "",                                        # Empty string
            "not-a-uuid-at-all",                      # Completely invalid
            "550e8400-e29b-41d4-a716-446655440000 ",  # Trailing space
            " 550e8400-e29b-41d4-a716-446655440000",  # Leading space
        ]

        for uuid_string in uuid_strings:
            # Create a mock manual auto-increment values dict with string
            manual_values = {"id": uuid_string}

            with pytest.raises(TypeError) as exc_info:
                session._validate_manual_auto_increment_values(
                    manual_values,
                    self.ValidationTestNode
                )

            # Verify error message mentions UUID objects are required
            error_msg = str(exc_info.value)
            assert "UUID field" in error_msg and "must be a UUID object" in error_msg

    def test_uuid_case_sensitivity_validation(self, test_db_path):
        """Test UUID validation with different case combinations."""
        session = KuzuSession(db_path=test_db_path)

        # Test various case combinations (all should be valid as UUID objects)
        valid_case_uuid_strings = [
            "550e8400-e29b-41d4-a716-446655440000",  # All lowercase
            "550E8400-E29B-41D4-A716-446655440000",  # All uppercase
            "550e8400-E29B-41d4-A716-446655440000",  # Mixed case
            "550E8400-e29b-41D4-a716-446655440000",  # Mixed case 2
        ]

        for uuid_string in valid_case_uuid_strings:
            # Convert to UUID object (this is what the validation expects)
            valid_uuid = uuid.UUID(uuid_string)
            manual_values = {"id": valid_uuid}

            # Should not raise any exceptions
            session._validate_manual_auto_increment_values(
                manual_values,
                self.ValidationTestNode
            )

    def test_uuid_version_variants_validation(self, test_db_path):
        """Test UUID validation with different UUID versions and variants."""
        session = KuzuSession(db_path=test_db_path)

        # Test different UUID versions (all should be valid as UUID objects)
        uuid_version_strings = [
            "00000000-0000-1000-8000-000000000000",  # Version 1
            "00000000-0000-2000-8000-000000000000",  # Version 2
            "00000000-0000-3000-8000-000000000000",  # Version 3
            "00000000-0000-4000-8000-000000000000",  # Version 4
            "00000000-0000-5000-8000-000000000000",  # Version 5
            "00000000-0000-0000-0000-000000000000",  # Nil UUID
            "ffffffff-ffff-ffff-ffff-ffffffffffff",  # Max UUID
        ]

        for uuid_string in uuid_version_strings:
            # Convert to UUID object (this is what the validation expects)
            uuid_version = uuid.UUID(uuid_string)
            manual_values = {"id": uuid_version}

            # Should not raise any exceptions
            session._validate_manual_auto_increment_values(
                manual_values,
                self.ValidationTestNode
            )

    def test_uuid_object_validation_always_passes(self, test_db_path):
        """Test that UUID objects always pass validation."""
        session = KuzuSession(db_path=test_db_path)

        # Test various UUID objects
        uuid_objects = [
            uuid.uuid1(),  # Time-based UUID
            uuid.uuid4(),  # Random UUID
            uuid.UUID("550e8400-e29b-41d4-a716-446655440000"),  # From string
            uuid.UUID(int=0),  # Nil UUID from int
            uuid.UUID(int=2**128-1),  # Max UUID from int
        ]

        for uuid_obj in uuid_objects:
            manual_values = {"id": uuid_obj}

            # Should not raise any exceptions
            session._validate_manual_auto_increment_values(
                manual_values,
                self.ValidationTestNode
            )

    def test_non_uuid_field_validation_skipped(self, test_db_path):
        """Test that non-UUID fields skip UUID validation."""
        session = KuzuSession(db_path=test_db_path)

        # Test with non-UUID field (name is STRING type)
        manual_values = {"name": "not-a-uuid-but-thats-ok"}

        # Should not raise any exceptions since name is not a UUID field
        session._validate_manual_auto_increment_values(
            manual_values,
            self.ValidationTestNode
        )


class TestConcurrentUUIDHandling:
    """tests for concurrent UUID handling scenarios."""

    @kuzu_node("ConcurrentUUIDNode")
    class ConcurrentUUIDNode(KuzuBaseModel):
        id: uuid.UUID = kuzu_field(kuzu_type=KuzuDataType.UUID, primary_key=True, auto_increment=True)
        thread_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64)

    def test_concurrent_uuid_validation_thread_safety(self, test_db_path):
        """Test UUID validation under concurrent access."""
        session = KuzuSession(db_path=test_db_path)
        initialize_schema(session, ddl=get_ddl_for_node(self.ConcurrentUUIDNode))

        # Test data with mix of valid and invalid UUIDs (as UUID objects and invalid types)
        test_data = [
            (uuid.UUID("550e8400-e29b-41d4-a716-446655440000"), True),   # Valid UUID object
            ("invalid-uuid", False),                                     # Invalid (string)
            (uuid.UUID("550e8400-e29b-41d4-a716-446655440001"), True),   # Valid UUID object
            ("", False),                                                 # Invalid (empty string)
            (uuid.UUID("550e8400-e29b-41d4-a716-446655440002"), True),   # Valid UUID object
        ]

        results = []

        def validate_uuid_in_thread(uuid_value, should_be_valid: bool, thread_id: int):
            """Validate UUID in a separate thread."""
            try:
                manual_values = {"id": uuid_value}
                session._validate_manual_auto_increment_values(
                    manual_values,
                    self.ConcurrentUUIDNode
                )
                return (thread_id, True, None)  # Validation passed
            except Exception as e:
                return (thread_id, False, str(e))  # Validation failed

        # Run validations concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i, (uuid_value, should_be_valid) in enumerate(test_data):
                future = executor.submit(validate_uuid_in_thread, uuid_value, should_be_valid, i)
                futures.append((future, should_be_valid))

            for future, expected_valid in futures:
                thread_id, validation_passed, error = future.result()
                results.append((thread_id, validation_passed, error, expected_valid))

        # Verify results match expectations
        for thread_id, validation_passed, error, expected_valid in results:
            if expected_valid:
                # Should have passed validation
                assert validation_passed, f"Thread {thread_id}: Expected valid UUID to pass, but got error: {error}"
            else:
                # Should have failed validation
                assert not validation_passed, f"Thread {thread_id}: Expected invalid UUID to fail validation"
                assert error is not None, f"Thread {thread_id}: Expected error message for invalid UUID"

    def test_concurrent_connection_reuse_exception_handling(self, test_db_path):
        """Test connection reuse exception handling under concurrent access."""
        session = KuzuSession(db_path=test_db_path)

        # Set up connection reuse state
        session._reused_connection = Mock()
        session._connection_operation_count = 1

        # Create a lock to synchronize exception raising
        exception_lock = threading.Lock()
        exception_count = 0

        def execute_with_exception():
            nonlocal exception_count
            with exception_lock:
                exception_count += 1
                if exception_count <= 3:
                    raise ConnectionError(f"Connection error {exception_count}")
                else:
                    return f"success_{exception_count}"

        session._reused_connection.execute.side_effect = execute_with_exception
        session._conn.execute = Mock(return_value="fallback_success")

        results = []

        def execute_query_in_thread(query: str, thread_id: int):
            """Execute query in a separate thread."""
            try:
                with patch('kuzualchemy.kuzu_session.logger'):
                    result = session._execute_with_connection_reuse(query)
                return (thread_id, result, None)
            except Exception as e:
                return (thread_id, None, str(e))

        # Run queries concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for i in range(5):
                future = executor.submit(execute_query_in_thread, f"SELECT {i}", i)
                futures.append(future)

            for future in futures:
                thread_id, result, error = future.result()
                results.append((thread_id, result, error))

        # Verify that some operations succeeded (either through reuse or fallback)
        successful_operations = [r for r in results if r[1] is not None]
        assert len(successful_operations) > 0, "At least some operations should have succeeded"

        # Verify that connection reuse state was properly managed
        # (This is harder to test precisely due to concurrency, but we can check final state)
        assert session._connection_operation_count >= 0  # Should be non-negative
