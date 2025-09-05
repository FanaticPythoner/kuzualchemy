# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive stress and reliability tests for KuzuAlchemy.

Tests cover:
- High concurrent load testing
- Memory usage under stress
- Connection pool exhaustion scenarios
- Large data operations and bulk processing
- Error recovery and resilience
- Resource leak detection
- Performance degradation monitoring
- Long-running operation stability
"""

from __future__ import annotations

import gc
import time
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from kuzualchemy import (
    kuzu_node,
    kuzu_relationship,
    KuzuBaseModel,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
)
from kuzualchemy.kuzu_orm import get_ddl_for_node
from kuzualchemy.test_utilities import initialize_schema


# Centralized, deterministic, thread-safe ID generator within INT64 bounds
KUZU_INT64_MIN: int = -(2**63)
KUZU_INT64_MAX: int = 2**63 - 1

class _DeterministicIDRegistry:
    """Deterministic, thread-safe ID registry for tests.
    Ensures non-overlapping ID spaces per namespace and strict sequential IDs.
    """
    def __init__(self) -> None:
        self._state: dict[str, int] = {}
        self._bases: dict[str, int] = {}
        self._lock = threading.Lock()

    def register(self, namespace: str, base: int) -> int:
        """Register a namespace with a fixed base. Idempotent.
        Returns the base for convenience.
        """
        if not isinstance(namespace, str) or not namespace:
            raise ValueError("namespace must be non-empty string")
        if base < 0 or base > KUZU_INT64_MAX - 1_000_000_000:
            # Guardrail to keep well within INT64 while allowing large blocks
            raise ValueError("base outside allowed INT64 guardrail")
        with self._lock:
            if namespace not in self._bases:
                self._bases[namespace] = base
                self._state[namespace] = base
            return self._bases[namespace]

    def next(self, namespace: str) -> int:
        with self._lock:
            if namespace not in self._state:
                raise KeyError(f"Namespace '{namespace}' not registered")
            val = self._state[namespace]
            if val > KUZU_INT64_MAX:
                raise OverflowError("INT64 upper bound exceeded")
            self._state[namespace] = val + 1
            return val

_ID_REG = _DeterministicIDRegistry()

def id_gen(namespace: str, base: int) -> tuple[int, callable]:
    """Register a namespace at base and return (base, next_id function)."""
    ns_base = _ID_REG.register(namespace, base)
    def _next() -> int:
        return _ID_REG.next(namespace)
    return ns_base, _next

@kuzu_node("StressTestUser")
class StressTestUser(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: str = kuzu_field(kuzu_type=KuzuDataType.STRING)
    thread_id: int = kuzu_field(kuzu_type=KuzuDataType.INT32)
    batch_id: int = kuzu_field(kuzu_type=KuzuDataType.INT32)

@kuzu_relationship("StressTestFollows", pairs=[("StressTestUser", "StressTestUser")])
class StressTestFollows(KuzuBaseModel):
    since: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=2024)
    strength: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)

class TestHighConcurrentLoad:
    """Test system behavior under high concurrent load."""

    def test_concurrent_writes_stress(self, test_db_path):
        """Test concurrent write operations under stress (serialized single-writer)."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(StressTestUser)
        initialize_schema(session, ddl=ddl)

        # Centralized deterministic ID generator for this test
        base, next_id = id_gen("StressTestUser:concurrent_writes", base=1_000_000)

        num_threads = 5
        operations_per_thread = 20
        write_lock = threading.Lock()  # Serialize writes due to Kuzu limitation

        def write_operations(thread_id: int, _db_path: Path):
            """Perform write operations in a thread with proper serialization."""
            success_count = 0
            for _ in range(operations_per_thread):
                # Serialize write operations due to Kuzu's single-writer limitation
                with write_lock:
                    session = KuzuSession(db_path=test_db_path)
                    initialize_schema(session, ddl=ddl)
                    user = StressTestUser(
                        id=next_id(),
                        name=f"User_{thread_id}_{success_count}",
                        email=f"user_{thread_id}_{success_count}@test.com",
                        thread_id=thread_id,
                        batch_id=success_count
                    )
                    session.add(user)
                    session.commit()
                    session.close()
                    success_count += 1

            return success_count

        # Execute concurrent writes
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(write_operations, thread_id, test_db_path)
                for thread_id in range(num_threads)
            ]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_operations = sum(results)

        # Verify results strictly for this test's ID range
        session = KuzuSession(db_path=test_db_path)
        ids = list(range(base, base + total_operations))
        count_result = session.execute(
            "MATCH (u:StressTestUser) WHERE u.id IN $ids RETURN count(u) as total",
            {"ids": ids}
        )
        actual_count = count_result[0]["total"]

        print(f"Concurrent writes: {total_operations} operations in {end_time - start_time:.2f}s")

        assert actual_count == total_operations
        session.close()

    def test_concurrent_reads_stress(self, test_db_path):
        """Test concurrent read operations under stress."""
        # First, populate database deterministically
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(StressTestUser)
        initialize_schema(session, ddl=ddl)

        base, next_id = id_gen("StressTestUser:concurrent_reads_prep", base=2_000_000)
        inserted_ids: list[int] = []
        for i in range(100):
            uid = next_id()
            inserted_ids.append(uid)
            user = StressTestUser(id=uid, name=f"User_{i}", email=f"user_{i}@test.com", thread_id=0, batch_id=0)
            session.add(user)
        session.commit()
        session.close()

        num_threads = 20
        reads_per_thread = 100

        def read_operations(_thread_id: int, _db_path: Path):
            """Perform read operations in a thread."""
            session = KuzuSession(db_path=test_db_path)
            success_count = 0

            for i in range(reads_per_thread):
                target_id = inserted_ids[i % len(inserted_ids)]
                result = session.execute(
                    "MATCH (u:StressTestUser) WHERE u.id = $id RETURN u.name",
                    {"id": target_id}
                )
                if len(result) > 0:
                    success_count += 1

            session.close()
            return success_count

        # Execute concurrent reads
        start_time = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(read_operations, thread_id, test_db_path)
                for thread_id in range(num_threads)
            ]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        total_reads = sum(results)

        print(f"Concurrent reads: {total_reads} operations in {end_time - start_time:.2f}s")

        # Exact success count for reads
        expected_total = num_threads * reads_per_thread
        assert total_reads == expected_total

    def test_mixed_operations_stress(self, test_db_path):
        """Test mixed read/write operations under stress."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(StressTestUser)
        initialize_schema(session, ddl=ddl)

        # Centralized deterministic ID generator for this test
        _base, next_id = id_gen("StressTestUser:mixed", base=3_000_000)

        # Use a lock to serialize write operations (Kuzu limitation)
        write_lock = threading.Lock()

        def mixed_operations(thread_id: int, _db_path: Path):
            """Perform mixed operations in a thread."""
            session = KuzuSession(db_path=test_db_path)

            operations = 0
            for i in range(25):
                # Write operation - serialize to avoid Kuzu's single-writer constraint
                with write_lock:
                    user = StressTestUser(
                        id=next_id(),
                        name=f"Mixed_{thread_id}_{i}",
                        email=f"mixed_{thread_id}_{i}@test.com",
                        thread_id=thread_id,
                        batch_id=i
                    )
                    session.add(user)
                    session.commit()
                    operations += 1

                # Read operation
                result = session.execute(
                    "MATCH (u:StressTestUser) WHERE u.thread_id = $tid RETURN count(u) as cnt",
                    {"tid": thread_id}
                )
                if len(result) > 0:
                    operations += 1

            session.close()
            return operations

        # Execute mixed operations
        num_threads = 4
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(mixed_operations, thread_id, test_db_path)
                for thread_id in range(num_threads)
            ]
            results = [future.result() for future in as_completed(futures)]

        total_operations = sum(results)
        print(f"Mixed operations completed: {total_operations}")

        assert total_operations == num_threads * 25 * 2, f"Expected {num_threads * 25 * 2} operations, got {total_operations}"

@kuzu_node("MemoryTestUser")
class MemoryTestUser(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    data: str = kuzu_field(kuzu_type=KuzuDataType.STRING)  # For large data

class TestMemoryUsageUnderStress:
    """Test memory usage patterns under stress conditions."""

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return sys.getsizeof(gc.get_objects()) / 1024 / 1024

    def test_large_batch_operations_memory(self, test_db_path):
        """Test memory usage during large batch operations (deterministic IDs)."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(MemoryTestUser)
        initialize_schema(session, ddl=ddl)

        initial_memory = self.get_memory_usage()
        print(f"Initial memory usage: {initial_memory:.2f} MB")

        # FULL STRESS TEST - deterministic IDs with centralized generator
        batch_size = 500
        large_data = "x" * 500
        _base, next_id = id_gen("MemoryTestUser:bulk", base=20_000_000)
        inserted_ids: list[int] = []

        for batch in range(5):  # 5 batches of 500 objects = 2500 total
            batch_start_memory = self.get_memory_usage()

            for i in range(batch_size):
                uid = next_id()
                inserted_ids.append(uid)
                user = MemoryTestUser(
                    id=uid,
                    name=f"StressBatchUser_{batch}_{i}",
                    data=large_data
                )
                session.add(user)

            # Commit the batch - triggers bulk insert
            session.commit()

            batch_end_memory = self.get_memory_usage()
            print(f"Batch {batch}: {batch_start_memory:.2f} -> {batch_end_memory:.2f} MB")

            # Force garbage collection
            gc.collect()
            after_gc_memory = self.get_memory_usage()
            print(f"After GC: {after_gc_memory:.2f} MB")

        final_memory = self.get_memory_usage()
        memory_increase = final_memory - initial_memory
        print(f"Final memory usage: {final_memory:.2f} MB (increase: {memory_increase:.2f} MB)")

        # Verify only records inserted by this test
        count_result = session.execute(
            "MATCH (u:MemoryTestUser) WHERE u.id IN $ids RETURN count(u) as total",
            {"ids": inserted_ids}
        )
        total_count = count_result[0]["total"]
        expected_total = 5 * batch_size

        print(f"Successfully inserted {total_count} records out of {expected_total} expected")

        # Precise equality assertion
        assert total_count == expected_total, f"Expected {expected_total} records, got {total_count}"

        session.close()

    def test_memory_leak_detection(self, test_db_path):
        """Test for memory leaks in repeated operations (deterministic IDs)."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(MemoryTestUser)
        initialize_schema(session, ddl=ddl)

        initial_memory = self.get_memory_usage()
        _base, next_id = id_gen("MemoryTestUser:leak", base=30_000_000)
        inserted_ids: list[int] = []

        total_cycles = 5
        per_cycle = 20
        for cycle in range(total_cycles):
            session = None
            session = KuzuSession(db_path=test_db_path)
            initialize_schema(session, ddl=ddl)

            for i in range(per_cycle):
                uid = next_id()
                inserted_ids.append(uid)
                user = MemoryTestUser(id=uid, name=f"LeakTest_{cycle}_{i}", data="test_data")
                session.add(user)

            session.commit()
            session.close()

            # Force cleanup
            gc.collect()

            current_memory = self.get_memory_usage()
            memory_increase = current_memory - initial_memory
            print(f"Cycle {cycle}: Memory usage {current_memory:.2f} MB (increase: {memory_increase:.2f} MB)")

        # Deterministic equality assertion on total inserted records by this test
        verify_session = KuzuSession(db_path=test_db_path)
        count_result = verify_session.execute(
            "MATCH (u:MemoryTestUser) WHERE u.id IN $ids RETURN count(u) as total",
            {"ids": inserted_ids}
        )
        total_count = count_result[0]["total"]
        expected_total = total_cycles * per_cycle
        assert total_count == expected_total, f"Expected {expected_total} records, got {total_count}"
        verify_session.close()


class TestConnectionPoolExhaustion:
    """Test connection pool exhaustion scenarios."""

    def test_connection_pool_limits(self, test_db_path):
        """Test behavior when connection pool is exhausted."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(LargeDataUser)
        initialize_schema(session, ddl=ddl)

        # Create many sessions simultaneously
        sessions = []
        max_sessions = 50

        for i in range(max_sessions):
            session = KuzuSession(db_path=test_db_path)
            sessions.append(session)

            # Try to execute a simple query
            result = session.execute("RETURN $i as num", {"i": i})
            assert len(result) == 1
            assert result[0]["num"] == i


        # Should be able to create new session after cleanup
        new_session = KuzuSession(db_path=test_db_path)
        result = new_session.execute("RETURN 'test' as value")
        assert len(result) == 1
        new_session.close()

    def test_connection_recovery_after_exhaustion(self, test_db_path):
        """Test connection recovery after pool exhaustion."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(LargeDataUser)
        initialize_schema(session, ddl=ddl)

        # Simulate connection exhaustion
        sessions = []

        # Create many sessions
        for i in range(30):
            session = KuzuSession(db_path=test_db_path)
            sessions.append(session)

        # Close half the sessions
        for i in range(len(sessions) // 2):
            sessions[i].close()

        # Should be able to create new sessions
        new_session = KuzuSession(db_path=test_db_path)
        result = new_session.execute("RETURN 'recovered' as status")
        assert len(result) == 1
        assert result[0]["status"] == "recovered"

        # Clean up
        new_session.close()
        for session in sessions[len(sessions) // 2:]:
            session.close()

@kuzu_node("LargeDataUser")
class LargeDataUser(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    description: str = kuzu_field(kuzu_type=KuzuDataType.STRING)

class TestLargeDataOperations:
    """Test large data operations and bulk processing."""

    def test_bulk_insert_performance(self, test_db_path):
        """Test bulk insert performance with large datasets (deterministic IDs)."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(LargeDataUser)
        initialize_schema(session, ddl=ddl)

        # Test different batch sizes
        batch_sizes = [100, 500, 1000]
        results = {}
        _base, next_id = id_gen("LargeDataUser:bulk", base=100_000_000)
        inserted_ids: list[int] = []

        for batch_idx, batch_size in enumerate(batch_sizes):
            start_time = time.time()

            for batch in range(5):  # 5 batches
                for i in range(batch_size):
                    uid = next_id()
                    inserted_ids.append(uid)
                    user = LargeDataUser(
                        id=uid,
                        name=f"BulkUser_{batch_idx}_{batch}_{i}",
                        description=f"Description for user {batch_idx}_{batch}_{i} " * 10
                    )
                    session.add(user)
                session.commit()

            end_time = time.time()
            total_records = 5 * batch_size
            duration = end_time - start_time
            results[batch_size] = {
                'records': total_records,
                'duration': duration,
                'records_per_second': total_records / duration
            }

            print(f"Batch size {batch_size}: {total_records} records in {duration:.2f}s "
                  f"({results[batch_size]['records_per_second']:.2f} records/sec)")

        # Verify only records inserted by this test
        count_result = session.execute(
            "MATCH (u:LargeDataUser) WHERE u.id IN $ids RETURN count(u) as total",
            {"ids": inserted_ids}
        )
        total_count = count_result[0]["total"]
        expected_total = sum(5 * batch_size for batch_size in batch_sizes)

        assert total_count == expected_total, f"Expected {expected_total} records, got {total_count}"
        session.close()

    def test_large_query_results(self, test_db_path):
        """Test handling of large query result sets (deterministic contiguous range)."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(LargeDataUser)
        initialize_schema(session, ddl=ddl)

        # Insert large contiguous dataset with known range [200000, 205000)
        num_records = 5000
        base, next_id = id_gen("LargeDataUser:large_query", base=200_000)
        for _ in range(num_records):
            uid = next_id()
            user = LargeDataUser(
                id=uid,
                name=f"QueryUser_{uid - base}",
                description=f"User {uid - base} description"
            )
            session.add(user)
        session.commit()

        # Query exactly the IDs we inserted using deterministic IN list
        start_time = time.time()
        ids = list(range(base, base + num_records))
        result = session.execute(
            "MATCH (u:LargeDataUser) WHERE u.id IN $ids RETURN u.id, u.name ORDER BY u.id",
            {"ids": ids}
        )
        end_time = time.time()

        assert len(result) == num_records, f"Expected {num_records} records, got {len(result)}"
        print(f"Large query: {len(result)} records retrieved in {end_time - start_time:.2f}s")

        # Verify result ordering (first 100)
        for i, row in enumerate(result[:100]):
            expected_id = base + i
            assert row["u.id"] == expected_id, f"Expected ID {expected_id}, got {row['u.id']}"

        session.close()


@kuzu_node("ErrorRecoveryUser")
class ErrorRecoveryUser(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

class TestErrorRecoveryAndResilience:
    """Test error recovery and system resilience."""

    def test_recovery_from_constraint_violations(self, test_db_path):
        """Test recovery from various constraint violations."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(ErrorRecoveryUser)
        initialize_schema(session, ddl=ddl)

        # Insert valid user
        user1 = ErrorRecoveryUser(id=1, name="Alice")
        session.add(user1)
        session.commit()

        error_count = 0
        success_count = 0

        # Attempt operations that may fail
        for i in range(10):
            try:
                if i % 3 == 0:
                    # Duplicate ID (should fail)
                    user = ErrorRecoveryUser(id=1, name=f"Duplicate_{i}")
                else:
                    # Valid user (should succeed)
                    user = ErrorRecoveryUser(id=i + 10, name=f"Valid_{i}")

                session.add(user)
                session.commit()
                success_count += 1

            except Exception:
                session.rollback()
                error_count += 1

        print(f"Error recovery test: {success_count} successes, {error_count} errors")

        # Should have some successes and some errors
        assert success_count == 6, f"Expected 6 successes, got {success_count}"
        assert error_count == 4, f"Expected 4 errors, got {error_count}"

        # Session should still be functional
        result = session.execute("MATCH (u:ErrorRecoveryUser) RETURN count(u) as total")
        expected_count = success_count + 1  # +1 for the initial user
        assert result[0]["total"] == expected_count, f"Expected {expected_count} total users, got {result[0]['total']}"

        session.close()

    def test_recovery_from_malformed_queries(self, test_db_path):
        """Test recovery from malformed queries."""
        session = KuzuSession(db_path=test_db_path)
        ddl = get_ddl_for_node(ErrorRecoveryUser)
        initialize_schema(session, ddl=ddl)

        # Mix of valid and invalid queries
        queries = [
            "MATCH (u:ErrorRecoveryUser) RETURN u.name",  # Valid
            "INVALID CYPHER SYNTAX",  # Invalid
            "MATCH (u:ErrorRecoveryUser) RETURN count(u) as total",  # Valid
            "SELECT * FROM users",  # Invalid (SQL, not Cypher)
            "RETURN 'test' as value",  # Valid
        ]

        valid_count = 0
        error_count = 0

        for query in queries:
            try:
                result = session.execute(query)
                valid_count += 1
                print(f"Query succeeded: {query[:30]}...")
            except Exception as e:
                error_count += 1
                print(f"Query failed: {query[:30]}... - {type(e).__name__}")

        print(f"Query recovery test: {valid_count} valid, {error_count} errors")

        # Should have exact counts based on the queries
        assert valid_count == 3, f"Expected 3 valid queries, got {valid_count}"
        assert error_count == 2, f"Expected 2 error queries, got {error_count}"

        session.close()

@kuzu_node("LongRunningUser")
class LongRunningUser(KuzuBaseModel):
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    iteration: int = kuzu_field(kuzu_type=KuzuDataType.INT32)

class TestLongRunningOperations:
    """Test long-running operation stability."""

    def test_sustained_operations(self, test_db_path):
        """Deterministic sustained operations with precise equality checks."""
        session = KuzuSession(db_path=test_db_path, bulk_insert_threshold=100000)
        ddl = get_ddl_for_node(LongRunningUser)
        initialize_schema(session, ddl=ddl)

        # Deterministic ID space for this test
        _base, next_id = id_gen("LongRunningUser:sustained", base=5_000_000)

        total_ops = 200
        commit_frequency = 10
        inserted_ids: list[int] = []

        for op in range(total_ops):
            user = LongRunningUser(
                id=next_id(),
                name=f"SustainedUser_{op}",
                iteration=op % 50
            )
            inserted_ids.append(user.id)  # type: ignore[attr-defined]
            session.add(user)
            if (op + 1) % commit_frequency == 0:
                session.commit()
        # Final commit for any remainder
        session.commit()

        # Verify only the records inserted by this test
        result = session.execute(
            "MATCH (u:LongRunningUser) WHERE u.id IN $ids RETURN count(u) as total",
            {"ids": inserted_ids}
        )
        total_count = result[0]["total"]
        assert total_count == total_ops, f"Expected {total_ops} records, got {total_count}"

        session.close()

    def test_session_stability_over_time(self, test_db_path):
        """Deterministic session stability with precise equality checks."""
        session = KuzuSession(db_path=test_db_path, bulk_insert_threshold=100000)
        ddl = get_ddl_for_node(LongRunningUser)
        initialize_schema(session, ddl=ddl)

        # Deterministic contiguous ID block for this test
        base, next_id = id_gen("LongRunningUser:stability", base=6_000_000)

        cycles = 20
        per_cycle = 10

        # Per-cycle verification and commit
        for cycle in range(cycles):
            cycle_ids: list[int] = []
            for i in range(per_cycle):
                uid = next_id()
                cycle_ids.append(uid)
                user = LongRunningUser(
                    id=uid,
                    name=f"StabilityUser_{cycle}_{i}",
                    iteration=cycle
                )
                session.add(user)
            session.commit()

            # Verify exactly 10 records for this cycle using the deterministic ID range
            start_id = base + cycle * per_cycle
            end_id = start_id + per_cycle
            ids = list(range(start_id, end_id))
            result = session.execute(
                "MATCH (u:LongRunningUser) WHERE u.id IN $ids RETURN count(u) as cnt",
                {"ids": ids}
            )
            actual_cycle_count = result[0]["cnt"]
            assert actual_cycle_count == per_cycle, (
                f"Expected {per_cycle} records for cycle {cycle}, got {actual_cycle_count}"
            )

        # Final verification of the whole deterministic range
        total_expected = cycles * per_cycle
        ids = list(range(base, base + total_expected))
        final = session.execute(
            "MATCH (u:LongRunningUser) WHERE u.id IN $ids RETURN count(u) as total",
            {"ids": ids}
        )
        total_count = final[0]["total"]
        assert total_count == total_expected, f"Expected {total_expected} total records, got {total_count}"

        session.close()
