# SPDX-FileCopyrightText: 2025 FanaticPythoner
# SPDX-License-Identifier: Apache-2.0

"""
Test utilities for KuzuAlchemy ORM.

Utilities for test execution with proper error handling.
"""

from __future__ import annotations

from typing import List, Optional, Set
import logging

from .kuzu_session import KuzuSession
from .kuzu_orm import get_all_ddl
from .constants import SessionOperationConstants

# @@ STEP: Configure logging for test utilities
logger = logging.getLogger(__name__)


class DDLExecutor:
    """
    Centralized DDL execution utility with proper error handling.
    
    :class: DDLExecutor
    :synopsis: Handles DDL statement execution with idempotent table creation
    """
    
    def __init__(self, session: KuzuSession):
        """
        Initialize DDL executor.
        
        :param session: KuzuSession instance to execute DDL on
        """
        self.session = session
        self._executed_tables: Set[str] = set()
        self._executed_relationships: Set[str] = set()
    
    def execute_all_ddl(self, ddl: Optional[str] = None, idempotent: bool = True) -> None:
        """
        Execute all DDL statements with proper error handling.
        
        :param ddl: Optional DDL string. If None, uses get_all_ddl()
        :param idempotent: If True, ignores "already exists" errors
        :raises RuntimeError: If non-idempotent error occurs
        """
        # @@ STEP: Get DDL if not provided
        if ddl is None:
            ddl = get_all_ddl()
        
        # @@ STEP: Skip if no DDL to execute
        if not ddl or not ddl.strip():
            logger.debug(SessionOperationConstants.NO_DDL_STATEMENTS)
            return
        
        # @@ STEP: Parse DDL statements
        statements = self._parse_ddl_statements(ddl)
        
        # @@ STEP: Execute each statement with proper error handling
        for statement in statements:
            self._execute_single_statement(statement, idempotent)
    
    def _parse_ddl_statements(self, ddl: str) -> List[str]:
        """
        Parse DDL string into individual statements.
        
        :param ddl: DDL string containing multiple statements
        :return: List of cleaned DDL statements
        """
        # @@ STEP: Split by semicolon and clean
        statements = []
        for stmt in ddl.split(';'):
            cleaned = stmt.strip()
            if cleaned and not cleaned.startswith('--'):  # Skip comments
                statements.append(cleaned)
        return statements
    
    def _execute_single_statement(self, statement: str, idempotent: bool) -> None:
        """
        Execute a single DDL statement with error handling.
        
        :param statement: DDL statement to execute
        :param idempotent: If True, ignores "already exists" errors
        :raises RuntimeError: If non-idempotent error occurs
        """
        # @@ STEP: Extract entity name and type from statement
        entity_type, entity_name = self._extract_entity_info(statement)
        
        # @@ STEP: Check if already executed to avoid redundant attempts
        if entity_type == SessionOperationConstants.NODE_ENTITY:
            if entity_name in self._executed_tables:
                logger.debug(SessionOperationConstants.SKIPPING_CREATED_NODE.format(entity_name))
                return
        elif entity_type == SessionOperationConstants.REL_ENTITY:
            if entity_name in self._executed_relationships:
                logger.debug(SessionOperationConstants.SKIPPING_CREATED_REL.format(entity_name))
                return
        
        # @@ STEP: Execute statement with specific error handling
        try:
            self.session.execute(statement)
            
            # || S.1: Track successful execution
            if entity_type == SessionOperationConstants.NODE_ENTITY:
                self._executed_tables.add(entity_name)
                logger.debug(SessionOperationConstants.CREATED_NODE_TABLE.format(entity_name))
            elif entity_type == SessionOperationConstants.REL_ENTITY:
                self._executed_relationships.add(entity_name)
                logger.debug(SessionOperationConstants.CREATED_REL_TABLE.format(entity_name))
                
        except RuntimeError as e:
            error_msg = str(e)
            
            # || S.2: Handle specific Kuzu errors
            if SessionOperationConstants.ALREADY_EXISTS_PATTERN in error_msg:
                if idempotent:
                    # || S.2.1: Track as already existing
                    if entity_type == SessionOperationConstants.NODE_ENTITY:
                        self._executed_tables.add(entity_name)
                        logger.debug(SessionOperationConstants.NODE_TABLE_EXISTS.format(entity_name))
                    elif entity_type == SessionOperationConstants.REL_ENTITY:
                        self._executed_relationships.add(entity_name)
                        logger.debug(SessionOperationConstants.REL_TABLE_EXISTS.format(entity_name))
                else:
                    # || S.2.2: Re-raise if not idempotent
                    raise RuntimeError(SessionOperationConstants.TABLE_ALREADY_EXISTS.format(entity_name)) from e
            
            elif SessionOperationConstants.BINDER_EXCEPTION_PATTERN in error_msg:
                # || S.3: Other binder exceptions should be re-raised
                raise RuntimeError(f"DDL execution failed: {error_msg}") from e
            
            else:
                # || S.4: Unknown RuntimeError - re-raise
                raise
    
    def _extract_entity_info(self, statement: str) -> tuple[str, str]:
        """
        Extract entity type and name from DDL statement.
        
        :param statement: DDL statement
        :return: Tuple of (entity_type, entity_name)
        """
        statement_upper = statement.upper()
        
        # @@ STEP: Determine entity type
        if "CREATE NODE TABLE" in statement_upper:
            entity_type = "NODE"
            # || S.1: Extract node table name
            parts = statement.split()
            for i, part in enumerate(parts):
                if part.upper() == "TABLE" and i + 1 < len(parts):
                    entity_name = parts[i + 1].split('(')[0].strip()
                    return entity_type, entity_name
        
        elif "CREATE REL TABLE" in statement_upper:
            entity_type = "REL"
            # || S.2: Extract relationship table name
            parts = statement.split()
            for i, part in enumerate(parts):
                if part.upper() == "TABLE" and i + 1 < len(parts):
                    entity_name = parts[i + 1].split('(')[0].strip()
                    return entity_type, entity_name
        
        # || S.3: Unrecognized DDL statement - FAIL EXPLICITLY
        raise ValueError(
            f"Unrecognized DDL statement type. Statement must contain either "
            f"'CREATE NODE TABLE' or 'CREATE REL TABLE'. Got: {statement[:100]}..."
        )
    
    def reset_tracking(self) -> None:
        """Reset internal tracking of executed tables."""
        self._executed_tables.clear()
        self._executed_relationships.clear()


def initialize_schema(session: KuzuSession, ddl: Optional[str] = None) -> None:
    """
    Initialize database schema with proper error handling.
    
    Convenience function for test setup.
    
    :param session: KuzuSession instance
    :param ddl: Optional DDL string. If None, uses get_all_ddl()
    """
    executor = DDLExecutor(session)
    executor.execute_all_ddl(ddl, idempotent=True)


def execute_ddl_safe(session: KuzuSession, statement: str, ignore_exists: bool = True) -> bool:
    """
    Execute a single DDL statement safely.
    
    :param session: KuzuSession instance
    :param statement: DDL statement to execute
    :param ignore_exists: If True, ignores "already exists" errors
    :return: True if executed successfully, False if already exists
    :raises RuntimeError: For non-idempotent errors
    """
    try:
        session.execute(statement)
        return True
    except RuntimeError as e:
        if SessionOperationConstants.ALREADY_EXISTS_PATTERN in str(e):
            if ignore_exists:
                return False
            raise
        # Re-raise other RuntimeErrors
        raise
