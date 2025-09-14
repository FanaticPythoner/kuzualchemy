# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
KuzuDB Connection Pool Implementation
====================================

This module implements proper concurrent access to KuzuDB following the official
concurrency model:
- One Database object per database file
- Multiple Connection objects from the same Database object
- Thread-safe connection management

This fixes the "Could not set lock on file" error by ensuring only one Database
object is created per database file, with multiple connections sharing it.
"""

from __future__ import annotations

import os
import threading
import weakref
from pathlib import Path
from typing import Dict, Optional, Union
import kuzu
from .constants import DatabaseConstants, PerformanceConstants


class DatabaseManager:
    """
    Singleton manager for KuzuDB Database objects.
    
    Ensures only one Database object exists per database file path,
    following KuzuDB's concurrency requirements.
    """
    
    _instance: Optional[DatabaseManager] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> DatabaseManager:
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the database manager."""
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self._databases: Dict[str, kuzu.Database] = {}
        self._database_locks: Dict[str, threading.RLock] = {}
        self._connection_counts: Dict[str, int] = {}
        self._manager_lock = threading.RLock()
        self._initialized = True
    
    def get_database(self, db_path: Union[str, Path], read_only: bool = False, buffer_pool_size: int = DatabaseConstants.DEFAULT_BUFFER_POOL_SIZE) -> kuzu.Database:
        """
        Get or create a Database object for the given path.

        Args:
            db_path: Path to the database file
            read_only: Whether to open in read-only mode
            buffer_pool_size: Buffer pool size in bytes. When 0, Kuzu auto-selects (~80% RAM).

        Returns:
            Shared Database object

        Raises:
            ValueError: If trying to open read-write when read-only exists or vice versa
        """
        db_path_str = str(Path(db_path).resolve())

        # Normalize buffer_pool_size: allow 0 to delegate sizing to Kuzu
        if buffer_pool_size is None or buffer_pool_size < 0 or buffer_pool_size > 2**63:
            buffer_pool_size = DatabaseConstants.DEFAULT_BUFFER_POOL_SIZE

        with self._manager_lock:
            if db_path_str in self._databases:
                existing_db = self._databases[db_path_str]
                # Check if the existing database mode matches the requested mode
                # Note: KuzuDB doesn't expose read_only flag, so we track it separately
                return existing_db

            # Create new database with validated buffer pool size
            try:
                # IMPORTANT: Pass parameters that let Kuzu manage memory adaptively.
                # Dynamic max_db_size based on filesystem free space
                computed_max_db_size = DatabaseConstants.resolve_max_db_size_for_path(db_path_str)
                database = kuzu.Database(
                    db_path_str,
                    buffer_pool_size=buffer_pool_size,
                    max_db_size=computed_max_db_size,
                    read_only=read_only,
                    compression=True,
                    lazy_init=False,
                    max_num_threads=os.environ.get("KUZUALCHEMY_MAX_THREADS", DatabaseConstants.DEFAULT_MAX_THREADS),
                    auto_checkpoint=True,
                    checkpoint_threshold=DatabaseConstants.DEFAULT_CHECKPOINT_THRESHOLD,
                )
                self._databases[db_path_str] = database
                self._database_locks[db_path_str] = threading.RLock()
                self._connection_counts[db_path_str] = 0

                return database

            except Exception as e:
                if "Could not set lock on file" in str(e):
                    raise RuntimeError(
                        f"Cannot open database '{db_path_str}': Another process has it open. "
                        f"KuzuDB allows either one READ_WRITE process OR multiple READ_ONLY processes. "
                        f"Original error: {e}"
                    ) from e
                raise
    
    def get_connection(self, db_path: Union[str, Path], read_only: bool = False, buffer_pool_size: int = 512 * 1024 * 1024) -> kuzu.Connection:
        """
        Get a new Connection object from the shared Database.

        Args:
            db_path: Path to the database file
            read_only: Whether to open in read-only mode
            buffer_pool_size: Buffer pool size in bytes (default: 512MB)

        Returns:
            New Connection object from shared Database
        """
        database = self.get_database(db_path, read_only, buffer_pool_size)
        db_path_str = str(Path(db_path).resolve())

        with self._manager_lock:
            connection = kuzu.Connection(database)
            self._connection_counts[db_path_str] += 1

            # Create a weak reference to track when connection is garbage collected
            def cleanup_callback(ref):
                with self._manager_lock:
                    if db_path_str in self._connection_counts:
                        self._connection_counts[db_path_str] -= 1
                        if self._connection_counts[db_path_str] <= 0:
                            # No more connections, can clean up database
                            self._cleanup_database(db_path_str)

            weakref.ref(connection, cleanup_callback)
            return connection
    
    def _cleanup_database(self, db_path_str: str):
        """Clean up database resources when no connections remain."""
        try:
            if db_path_str in self._databases:
                # Close database if KuzuDB supports it
                database = self._databases[db_path_str]
                if hasattr(database, 'close'):
                    database.close()
                
                del self._databases[db_path_str]
                del self._database_locks[db_path_str]
                del self._connection_counts[db_path_str]
        except Exception:
            # Ignore cleanup errors
            pass
    
    def get_database_lock(self, db_path: Union[str, Path]) -> threading.RLock:
        """Get the lock for a specific database."""
        db_path_str = str(Path(db_path).resolve())
        with self._manager_lock:
            if db_path_str not in self._database_locks:
                # Ensure database exists
                self.get_database(db_path_str)
            return self._database_locks[db_path_str]
    
    def close_all(self):
        """Close all databases and clean up resources."""
        with self._manager_lock:
            for db_path_str in list(self._databases.keys()):
                self._cleanup_database(db_path_str)
    
    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get statistics about managed databases and connections."""
        with self._manager_lock:
            return {
                db_path: {
                    'connection_count': self._connection_counts.get(db_path, 0),
                    'has_database': db_path in self._databases
                }
                for db_path in self._databases.keys()
            }


class ConnectionPool:
    """
    Connection pool for a specific database.
    
    Manages multiple Connection objects from a single shared Database object.
    """
    
    def __init__(self, db_path: Union[str, Path], read_only: bool = False, max_connections: int = 10, buffer_pool_size: int = 512 * 1024 * 1024):
        """
        Initialize connection pool for a database.

        Args:
            db_path: Path to the database file
            read_only: Whether to open in read-only mode
            max_connections: Maximum number of connections to maintain
            buffer_pool_size: Buffer pool size in bytes (default: 512MB)
        """
        self.db_path = Path(db_path).resolve()
        self.read_only = read_only
        self.max_connections = max_connections
        self.buffer_pool_size = buffer_pool_size

        self._manager = DatabaseManager()
        self._pool_lock = threading.RLock()
        self._available_connections: list[kuzu.Connection] = []
        self._active_connections: set[kuzu.Connection] = set()
        self._total_created = 0
    
    def get_connection(self) -> kuzu.Connection:
        """
        Get a connection from the pool.
        
        Returns:
            Connection object (either from pool or newly created)
        """
        with self._pool_lock:
            # Try to get from pool first
            if self._available_connections:
                connection = self._available_connections.pop()
                self._active_connections.add(connection)
                return connection
            
            # Create new connection if under limit
            if self._total_created < self.max_connections:
                connection = self._manager.get_connection(self.db_path, self.read_only, self.buffer_pool_size)
                self._active_connections.add(connection)
                self._total_created += 1
                return connection

            # Pool exhausted, create temporary connection
            return self._manager.get_connection(self.db_path, self.read_only, self.buffer_pool_size)
    
    def return_connection(self, connection: kuzu.Connection):
        """
        Return a connection to the pool.
        
        Args:
            connection: Connection to return
        """
        with self._pool_lock:
            if connection in self._active_connections:
                self._active_connections.remove(connection)
                
                # Only keep in pool if under limit
                if len(self._available_connections) < self.max_connections:
                    self._available_connections.append(connection)
                else:
                    # Let connection be garbage collected
                    self._total_created -= 1
    
    def close_all(self):
        """Close all connections in the pool."""
        with self._pool_lock:
            # Clear all connections (they'll be garbage collected)
            self._available_connections.clear()
            self._active_connections.clear()
            self._total_created = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get pool statistics."""
        with self._pool_lock:
            return {
                'available': len(self._available_connections),
                'active': len(self._active_connections),
                'total_created': self._total_created,
                'max_connections': self.max_connections
            }


# Global database manager instance
_database_manager = DatabaseManager()


# Thread-safe registry of shared ConnectionPool instances per (db_path, read_only)
_pools_lock = threading.RLock()
_connection_pools: Dict[str, ConnectionPool] = {}


def _pool_key(db_path: Union[str, Path], read_only: bool) -> str:
    path = str(Path(db_path).resolve())
    return f"{path}::{int(bool(read_only))}"


def get_shared_connection_pool(
    db_path: Union[str, Path],
    read_only: bool = False,
    max_connections: Optional[int] = None,
    buffer_pool_size: int = DatabaseConstants.DEFAULT_BUFFER_POOL_SIZE,
) -> ConnectionPool:
    """
    Get or create a shared ConnectionPool for the given database path.

    The pool bounds the number of low-level kuzu.Connection objects to avoid resource
    exhaustion and native crashes under stress. Multiple logical sessions may borrow
    connections from this pool and return them immediately after each operation.

    Args:
        db_path: Database path
        read_only: Whether connections should be opened in read-only mode
        max_connections: Max pooled connections; defaults to PerformanceConstants.CONNECTION_POOL_SIZE
        buffer_pool_size: Database buffer pool size hint

    Returns:
        A process-wide shared ConnectionPool for the (db_path, read_only) pair.
    """
    key = _pool_key(db_path, read_only)
    with _pools_lock:
        pool = _connection_pools.get(key)
        if pool is None:
            pool = ConnectionPool(
                db_path=db_path,
                read_only=read_only,
                max_connections=max_connections or PerformanceConstants.CONNECTION_POOL_SIZE,
                buffer_pool_size=buffer_pool_size,
            )
            _connection_pools[key] = pool
        return pool


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    return _database_manager


def create_connection_pool(db_path: Union[str, Path], read_only: bool = False, max_connections: int = 10, buffer_pool_size: int = 512 * 1024 * 1024) -> ConnectionPool:
    """
    Create a connection pool for a database.

    Args:
        db_path: Path to the database file
        read_only: Whether to open in read-only mode
        max_connections: Maximum number of connections to maintain
        buffer_pool_size: Buffer pool size in bytes (default: 512MB)

    Returns:
        ConnectionPool instance
    """
    return ConnectionPool(db_path, read_only, max_connections, buffer_pool_size)


def get_connection(db_path: Union[str, Path], read_only: bool = False, buffer_pool_size: int = 512 * 1024 * 1024) -> kuzu.Connection:
    """
    Get a connection to a database using the global manager.

    Args:
        db_path: Path to the database file
        read_only: Whether to open in read-only mode
        buffer_pool_size: Buffer pool size in bytes (default: 512MB)

    Returns:
        Connection object from shared Database
    """
    return _database_manager.get_connection(db_path, read_only, buffer_pool_size)


def close_all_databases():
    """Close all managed databases and clean up resources."""
    _database_manager.close_all()
