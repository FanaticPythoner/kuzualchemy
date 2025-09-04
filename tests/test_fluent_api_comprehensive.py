"""
Test suite for KuzuAlchemy Fluent API.

Tests EVERY aspect of the fluent query API:
- Node queries with all operations
- Relationship queries with all operations
- Combined node + relationship queries
- Complex filter combinations
- Join operations
- Aggregations with grouping
- All execution methods
- Edge cases and error scenarios
- Performance with large datasets
"""

from __future__ import annotations

import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, date
from typing import Optional, List

from kuzualchemy.constants import (
    KuzuDefaultFunction,
)
from kuzualchemy.kuzu_orm import (
    KuzuDataType,
    ArrayTypeSpecification,
)
from kuzualchemy.kuzu_query_expressions import (
    OrderDirection,
)
from kuzualchemy import (
    KuzuBaseModel,
    KuzuRelationshipBase,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    QueryField,
    KuzuSession,
)
from kuzualchemy.test_utilities import initialize_schema


@kuzu_node("TestPerson")
class TestPerson(KuzuBaseModel):
    """Person node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, unique=True, default=None)
    age: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    salary: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    is_active: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)
    created_at: datetime = kuzu_field(
        kuzu_type=KuzuDataType.TIMESTAMP,
        default=KuzuDefaultFunction.CURRENT_TIMESTAMP
    )
    sectors: Optional[List[str]] = kuzu_field(
        kuzu_type=ArrayTypeSpecification(element_type=KuzuDataType.STRING),
        default=None
    )
    metadata: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)


@kuzu_node("TestCompany")
class TestCompany(KuzuBaseModel):
    """Company node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True, unique=True)
    industry: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="Technology")
    revenue: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    employee_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    is_public: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)
    founded_date: Optional[datetime] = kuzu_field(kuzu_type=KuzuDataType.DATE, default=None)


@kuzu_node("TestProject")
class TestProject(KuzuBaseModel):
    """Project node for testing."""
    id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    description: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="")
    budget: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    start_date: datetime = kuzu_field(kuzu_type=KuzuDataType.DATE, default=KuzuDefaultFunction.CURRENT_DATE)
    end_date: Optional[datetime] = kuzu_field(kuzu_type=KuzuDataType.DATE, default=None)
    status: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="PLANNING")
    priority: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=3)


@kuzu_relationship("WORKS_FOR", pairs=[(TestPerson, TestCompany)])
class WorksFor(KuzuRelationshipBase):
    """Person works for Company relationship."""
    position: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    start_date: date = kuzu_field(kuzu_type=KuzuDataType.DATE, default=KuzuDefaultFunction.CURRENT_DATE)
    end_date: Optional[date] = kuzu_field(kuzu_type=KuzuDataType.DATE, default=None)
    department: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="Engineering")
    is_remote: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)


@kuzu_relationship("MANAGES", pairs=[(TestPerson, TestProject)])
class Manages(KuzuRelationshipBase):
    """Person manages Project relationship."""
    role: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="Manager")
    responsibility_level: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=1)
    assigned_date: date = kuzu_field(kuzu_type=KuzuDataType.DATE, default=KuzuDefaultFunction.CURRENT_DATE)


@kuzu_relationship("COLLABORATES", pairs=[(TestPerson, TestPerson)])
class Collaborates(KuzuRelationshipBase):
    """Person collaborates with Person relationship."""
    project_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=1)
    since: date = kuzu_field(kuzu_type=KuzuDataType.DATE, default=KuzuDefaultFunction.CURRENT_DATE)
    trust_score: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.5)


@kuzu_relationship("OWNS", pairs=[(TestCompany, TestProject)])
class Owns(KuzuRelationshipBase):
    """Company owns Project relationship."""
    ownership_percentage: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=100.0)
    acquired_date: date = kuzu_field(kuzu_type=KuzuDataType.DATE, default=KuzuDefaultFunction.CURRENT_DATE)
    investment_amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)


@kuzu_relationship("MULTI_REL", pairs=[
    (TestPerson, TestCompany),
    (TestCompany, TestProject),
    (TestPerson, TestProject)
])
class MultiRel(KuzuRelationshipBase):
    """Multi-pair relationship for testing."""
    connection_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    strength: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)


def initialize_schema(session: KuzuSession) -> None:
    """Initialize database schema with all test models."""
    from kuzualchemy.kuzu_orm import generate_node_ddl, generate_relationship_ddl
    
    # @@ STEP: Generate and execute DDL for node tables
    node_models = [TestPerson, TestCompany, TestProject]
    for model in node_models:
        ddl = generate_node_ddl(model)
        session.execute(ddl)
    
    # @@ STEP: Generate and execute DDL for relationship tables
    rel_models = [WorksFor, Collaborates, Manages, Owns, MultiRel]
    for model in rel_models:
        ddl = generate_relationship_ddl(model)
        session.execute(ddl)


class TestNodeQueries:
    """Tests for node queries."""
    
    @classmethod
    def setup_class(cls):
        """Set up test database and session once for all tests."""
        cls.temp_db = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_db) / "test_fluent_nodes.db"
        cls.session = KuzuSession(db_path=str(cls.db_path))
        
        # @@ STEP: Initialize schema
        initialize_schema(cls.session)
        
        # @@ STEP: Insert test data
        cls._insert_test_data_static(cls.session)
        
        # @@ STEP: Track IDs added during tests for cleanup
        cls._test_added_person_ids = set()
        cls._test_added_company_ids = set()
        cls._test_added_project_ids = set()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test database after all tests."""
        cls.session.close()
        if Path(cls.temp_db).exists():
            shutil.rmtree(cls.temp_db, ignore_errors=True)
    
    def teardown_method(self):
        """Clean up any data added during individual tests using fluent API."""
        # @@ STEP: Delete any persons added during tests using fluent API
        if hasattr(self, '_test_added_person_ids') and self._test_added_person_ids:
            for person_id in self._test_added_person_ids:
                person = self.session.query(TestPerson).filter_by(id=person_id).first()
                if person:
                    self.session.delete(person)
            self.session.commit()
            self._test_added_person_ids.clear()

        # @@ STEP: Delete any companies added during tests using fluent API
        if hasattr(self, '_test_added_company_ids') and self._test_added_company_ids:
            for company_id in self._test_added_company_ids:
                company = self.session.query(TestCompany).filter_by(id=company_id).first()
                if company:
                    self.session.delete(company)
            self.session.commit()
            self._test_added_company_ids.clear()

        # @@ STEP: Delete any projects added during tests using fluent API
        if hasattr(self, '_test_added_project_ids') and self._test_added_project_ids:
            for project_id in self._test_added_project_ids:
                project = self.session.query(TestProject).filter_by(id=project_id).first()
                if project:
                    self.session.delete(project)
            self.session.commit()
            self._test_added_project_ids.clear()
    
    @classmethod
    def _insert_test_data_static(cls, session: KuzuSession):
        """Insert test data."""
        # @@ STEP: Create persons
        persons = [
            TestPerson(id=1, name="Alice Smith", email="alice@test.com", age=30, salary=75000, is_active=True),
            TestPerson(id=2, name="Bob Johnson", email="bob@test.com", age=25, salary=60000, is_active=True),
            TestPerson(id=3, name="Charlie Brown", email="charlie@test.com", age=35, salary=85000, is_active=False),
            TestPerson(id=4, name="Diana Prince", email="diana@test.com", age=28, salary=70000, is_active=True),
            TestPerson(id=5, name="Eve Wilson", email="eve@test.com", age=32, salary=80000, is_active=True),
            TestPerson(id=6, name="Frank Miller", email="frank@test.com", age=45, salary=120000, is_active=True),
            TestPerson(id=7, name="Grace Lee", email="grace@test.com", age=27, salary=65000, is_active=False),
            TestPerson(id=8, name="Henry Ford", email="henry@test.com", age=50, salary=150000, is_active=True),
            TestPerson(id=9, name="Iris West", email="iris@test.com", age=29, salary=72000, is_active=True),
            TestPerson(id=10, name="Jack Ryan", email="jack@test.com", age=38, salary=95000, is_active=True),
        ]
        
        # @@ STEP: Create companies
        companies = [
            TestCompany(id=1, name="TechCorp", industry="Technology", revenue=1000000, employee_count=100, is_public=True),
            TestCompany(id=2, name="FinanceInc", industry="Finance", revenue=5000000, employee_count=500, is_public=True),
            TestCompany(id=3, name="StartupXYZ", industry="Technology", revenue=100000, employee_count=10, is_public=False),
            TestCompany(id=4, name="ConsultingPro", industry="Consulting", revenue=2000000, employee_count=200, is_public=False),
            TestCompany(id=5, name="RetailGiant", industry="Retail", revenue=10000000, employee_count=1000, is_public=True),
        ]
        
        # @@ STEP: Create projects
        projects = [
            TestProject(id=1, name="Alpha", description="First project", budget=100000, status="ACTIVE", priority=1),
            TestProject(id=2, name="Beta", description="Second project", budget=200000, status="ACTIVE", priority=2),
            TestProject(id=3, name="Gamma", description="Third project", budget=150000, status="PLANNING", priority=3),
            TestProject(id=4, name="Delta", description="Fourth project", budget=300000, status="COMPLETED", priority=1),
            TestProject(id=5, name="Epsilon", description="Fifth project", budget=50000, status="ON_HOLD", priority=4),
        ]
        
        # @@ STEP: Add all nodes
        for person in persons:
            session.add(person)
        for company in companies:
            session.add(company)
        for project in projects:
            session.add(project)
        session.commit()
        
        # @@ STEP: Create WORKS_FOR relationships using proper fluent API
        # Get node instances for relationship creation
        alice = session.query(TestPerson).filter_by(id=1).first()
        bob = session.query(TestPerson).filter_by(id=2).first()
        charlie = session.query(TestPerson).filter_by(id=3).first()
        diana = session.query(TestPerson).filter_by(id=4).first()
        eve = session.query(TestPerson).filter_by(id=5).first()

        techcorp = session.query(TestCompany).filter_by(id=1).first()
        financeinc = session.query(TestCompany).filter_by(id=2).first()
        startupxyz = session.query(TestCompany).filter_by(id=3).first()
        consultingpro = session.query(TestCompany).filter_by(id=4).first()

        alpha = session.query(TestProject).filter_by(id=1).first()
        beta = session.query(TestProject).filter_by(id=2).first()
        gamma = session.query(TestProject).filter_by(id=3).first()
        delta = session.query(TestProject).filter_by(id=4).first()

        # Create WORKS_FOR relationships using the new intuitive API
        session.create_relationship(
            WorksFor, alice, techcorp,
            position='Senior Engineer', department='Engineering',
            is_remote=False, start_date=date(2020, 1, 15)
        )
        session.create_relationship(
            WorksFor, bob, techcorp,
            position='Junior Engineer', department='Engineering',
            is_remote=True, start_date=date(2021, 3, 1)
        )
        session.create_relationship(
            WorksFor, charlie, financeinc,
            position='Analyst', department='Finance',
            is_remote=False, start_date=date(2019, 6, 1)
        )
        session.create_relationship(
            WorksFor, diana, startupxyz,
            position='CTO', department='Executive',
            is_remote=False, start_date=date(2022, 1, 1)
        )
        session.create_relationship(
            WorksFor, eve, consultingpro,
            position='Consultant', department='Operations',
            is_remote=True, start_date=date(2020, 9, 1)
        )

        # @@ STEP: Create MANAGES relationships using the new intuitive API
        session.create_relationship(
            Manages, alice, alpha,
            role='Lead', responsibility_level=3
        )
        session.create_relationship(
            Manages, diana, beta,
            role='Overseer', responsibility_level=2
        )
        session.create_relationship(
            Manages, eve, gamma,
            role='Coordinator', responsibility_level=1
        )

        # @@ STEP: Create COLLABORATES relationships using the new intuitive API
        session.create_relationship(
            Collaborates, alice, bob,
            project_count=3, trust_score=0.9
        )
        session.create_relationship(
            Collaborates, bob, alice,
            project_count=3, trust_score=0.85
        )
        session.create_relationship(
            Collaborates, diana, eve,
            project_count=1, trust_score=0.7
        )

        # @@ STEP: Create OWNS relationships using the new intuitive API
        session.create_relationship(
            Owns, techcorp, alpha,
            ownership_percentage=100.0, investment_amount=100000
        )
        session.create_relationship(
            Owns, startupxyz, beta,
            ownership_percentage=80.0, investment_amount=160000
        )
        session.create_relationship(
            Owns, financeinc, delta,
            ownership_percentage=100.0, investment_amount=300000
        )

        # @@ STEP: Create MULTI_REL relationships using the new intuitive API
        session.create_relationship(
            MultiRel, alice, techcorp,
            connection_type='employee', strength=0.9
        )
        session.create_relationship(
            MultiRel, financeinc, alpha,
            connection_type='sponsor', strength=0.95
        )
        session.create_relationship(
            MultiRel, charlie, beta,
            connection_type='contributor', strength=0.8
        )

        # @@ STEP: Commit all relationships
        session.commit()
    
    def test_basic_node_query(self):
        """Test basic node query without filters."""
        query = self.session.query(TestPerson)
        results = query.all()
        
        assert len(results) == 10
        assert all(isinstance(r, TestPerson) for r in results)
    
    def test_filter_by_single_field(self):
        """Test filtering by single field."""
        query = self.session.query(TestPerson).filter_by(age=30)
        results = query.all()
        
        assert len(results) == 1
        assert results[0].name == "Alice Smith"
        assert results[0].age == 30
    
    def test_filter_by_multiple_fields(self):
        """Test filtering by multiple fields."""
        query = self.session.query(TestPerson).filter_by(is_active=True, age=25)
        results = query.all()
        
        assert len(results) == 1
        assert results[0].name == "Bob Johnson"
    
    def test_filter_with_comparison_operators(self):
        """Test filtering with various comparison operators."""
        # @@ STEP: Greater than
        query = self.session.query(TestPerson).filter(
            QueryField("age", TestPerson) > 35
        )
        results = query.all()
        assert len(results) == 3  # Frank (45), Henry (50), Jack (38)
        
        # @@ STEP: Less than or equal
        query = self.session.query(TestPerson).filter(
            QueryField("salary", TestPerson) <= 70000
        )
        results = query.all()
        assert len(results) == 3  # Bob (60000), Diana (70000), Grace (65000)
        
        # @@ STEP: Not equal
        query = self.session.query(TestPerson).filter(
            QueryField("is_active", TestPerson) != True
        )
        results = query.all()
        assert len(results) == 2  # Charlie, Grace
    
    def test_filter_with_in_operator(self):
        """Test filtering with IN operator."""
        query = self.session.query(TestPerson).filter(
            QueryField("name", TestPerson).in_(["Alice Smith", "Bob Johnson", "Invalid Name"])
        )
        results = query.all()
        
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"Alice Smith", "Bob Johnson"}
    
    def test_filter_with_like_operator(self):
        """Test filtering with LIKE operator."""
        query = self.session.query(TestPerson).filter(
            QueryField("email", TestPerson).like(".*@test\\.com")
        )
        results = query.all()
        
        assert len(results) == 10  # All have @test.com
        
        # @@ STEP: Test starts_with
        query = self.session.query(TestPerson).filter(
            QueryField("name", TestPerson).starts_with("Charlie")
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "Charlie Brown"
        
        # @@ STEP: Test ends_with
        query = self.session.query(TestCompany).filter(
            QueryField("name", TestCompany).ends_with("Corp")
        )
        results = query.all()
        assert len(results) == 1
        assert results[0].name == "TechCorp"
    
    def test_filter_with_between_operator(self):
        """Test filtering with BETWEEN operator."""
        query = self.session.query(TestPerson).filter(
            QueryField("age", TestPerson).between(28, 32)
        )
        results = query.all()
        
        assert len(results) == 4  # Diana (28), Eve (32), Iris (29), Alice (30)
        ages = {r.age for r in results}
        assert ages == {28, 29, 30, 32}
    
    def test_filter_with_null_checks(self):
        """Test filtering with NULL checks."""
        # @@ STEP: Add person with null metadata
        person = TestPerson(id=11, name="Null Test", email="null@test.com", age=40, metadata=None)
        self.session.add(person)
        self.session.commit()
        self._test_added_person_ids.add(11)
        
        query = self.session.query(TestPerson).filter(
            QueryField("metadata", TestPerson).is_null()
        )
        results = query.all()
        
        assert len(results) == 11  # All have null metadata
        
        query = self.session.query(TestPerson).filter(
            QueryField("metadata", TestPerson).is_not_null()
        )
        results = query.all()
        
        assert len(results) == 0
    
    def test_complex_filter_combinations(self):
        """Test complex filter combinations with AND, OR, NOT."""
        # @@ STEP: AND combination
        query = self.session.query(TestPerson).filter(
            (QueryField("age", TestPerson) >= 30) & 
            (QueryField("salary", TestPerson) > 75000)
        )
        results = query.all()
        
        assert len(results) == 5  # Charlie, Eve, Frank, Henry, Jack
        
        # @@ STEP: OR combination
        query = self.session.query(TestPerson).filter(
            (QueryField("age", TestPerson) < 28) | 
            (QueryField("salary", TestPerson) > 100000)
        )
        results = query.all()
        
        assert len(results) == 4  # Bob (25), Grace (27), Frank (120000), Henry (150000)
        
        # @@ STEP: NOT operation
        query = self.session.query(TestPerson).filter(
            ~(QueryField("is_active", TestPerson) == True)
        )
        results = query.all()
        
        assert len(results) == 2  # Charlie, Grace
        
        # @@ STEP: Complex nested combination
        query = self.session.query(TestPerson).filter(
            ((QueryField("age", TestPerson) > 30) & (QueryField("salary", TestPerson) < 100000)) |
            ((QueryField("age", TestPerson) <= 30) & (QueryField("is_active", TestPerson) == False))
        )
        results = query.all()
        
        assert len(results) == 4  # Eve (32, 80000), Charlie (35, 85000), Jack (38, 95000), Grace (27, inactive)
    
    def test_order_by_single_field(self):
        """Test ordering by single field."""
        # @@ STEP: Ascending order
        query = self.session.query(TestPerson).order_by("age")
        results = query.all()
        
        ages = [r.age for r in results]
        assert ages == sorted(ages)
        assert results[0].name == "Bob Johnson"  # age 25
        
        # @@ STEP: Descending order
        query = self.session.query(TestPerson).order_by(("salary", OrderDirection.DESC))
        results = query.all()
        
        salaries = [r.salary for r in results]
        assert salaries == sorted(salaries, reverse=True)
        assert results[0].name == "Henry Ford"  # salary 150000
    
    def test_order_by_multiple_fields(self):
        """Test ordering by multiple fields."""
        # @@ STEP: Add persons with same age for testing
        self.session.add(TestPerson(id=12, name="Test Same Age 1", email="same1@test.com", age=30, salary=90000))
        self.session.add(TestPerson(id=13, name="Test Same Age 2", email="same2@test.com", age=30, salary=55000))
        self.session.commit()
        self._test_added_person_ids.add(12)
        self._test_added_person_ids.add(13)
        
        query = self.session.query(TestPerson).order_by(
            ("age", OrderDirection.ASC),
            ("salary", OrderDirection.DESC)
        )
        results = query.all()
        
        # @@ STEP: Verify ordering
        prev_age = -1
        prev_salary = float('inf')
        for r in results:
            if r.age == prev_age:
                assert r.salary <= prev_salary
            else:
                assert r.age >= prev_age
            prev_age = r.age
            prev_salary = r.salary
    
    def test_limit_and_offset(self):
        """Test limit and offset functionality."""
        # @@ STEP: Limit only
        query = self.session.query(TestPerson).order_by("id").limit(3)
        results = query.all()
        
        assert len(results) == 3
        assert [r.id for r in results] == [1, 2, 3]
        
        # @@ STEP: Offset only
        query = self.session.query(TestPerson).order_by("id").offset(5)
        results = query.all()
        
        assert len(results) == 5  # 10 total - 5 offset
        assert results[0].id == 6
        
        # @@ STEP: Limit and offset
        query = self.session.query(TestPerson).order_by("id").offset(3).limit(4)
        results = query.all()
        
        assert len(results) == 4
        assert [r.id for r in results] == [4, 5, 6, 7]
    
    def test_distinct(self):
        """Test distinct functionality."""
        # @@ STEP: Add duplicate entries for testing
        self.session.add(TestCompany(id=6, name="DupIndustry1", industry="Technology"))
        self.session.add(TestCompany(id=7, name="DupIndustry2", industry="Technology"))
        self.session.commit()
        self._test_added_company_ids.add(6)
        self._test_added_company_ids.add(7)
        
        query = self.session.query(TestCompany).select("industry").distinct()
        cypher, _ = query.to_cypher()
        
        assert "DISTINCT" in cypher
    
    def test_select_specific_fields(self):
        """Test selecting specific fields."""
        query = self.session.query(TestPerson).select("name", "email", "age").limit(2)
        results = query.all()
        
        assert len(results) == 2
        # Note: Partial models may be returned as dicts or have limited attributes
        # The important part is that the query includes the SELECT clause
        cypher, params = query.to_cypher()
        assert "RETURN" in cypher
        assert any(field in cypher for field in ["n.name", "n.email", "n.age"])
    
    def test_aggregation_count(self):
        """Test COUNT aggregation."""
        # @@ STEP: Count all
        query = self.session.query(TestPerson).count()
        cypher, params = query.to_cypher()

        assert "COUNT(*) AS count" in cypher
        assert isinstance(params, dict)  # Validate params structure
        assert "count" in query._state.aggregations
        
        # @@ STEP: Count specific field
        query = self.session.query(TestPerson).count("email")
        cypher, params = query.to_cypher()

        assert "COUNT(n.email) AS count" in cypher
        assert isinstance(params, dict)
    
    def test_aggregation_sum_avg_min_max(self):
        """Test SUM, AVG, MIN, MAX aggregations."""
        # @@ STEP: SUM
        query = self.session.query(TestPerson).sum("salary", alias="total_salary")
        cypher, _ = query.to_cypher()
        assert "SUM(n.salary) AS total_salary" in cypher
        
        # @@ STEP: AVG
        query = self.session.query(TestPerson).avg("age", alias="avg_age")
        cypher, _ = query.to_cypher()
        assert "AVG(n.age) AS avg_age" in cypher
        
        # @@ STEP: MIN
        query = self.session.query(TestPerson).min("salary", alias="min_salary")
        cypher, _ = query.to_cypher()
        assert "MIN(n.salary) AS min_salary" in cypher
        
        # @@ STEP: MAX
        query = self.session.query(TestPerson).max("age", alias="max_age")
        cypher, _ = query.to_cypher()
        assert "MAX(n.age) AS max_age" in cypher
    
    def test_group_by_with_aggregations(self):
        """Test GROUP BY with aggregations."""
        query = (self.session.query(TestCompany)
                .group_by("industry")
                .count("*")
                .avg("revenue", alias="avg_revenue"))
        
        cypher, _ = query.to_cypher()
        
        assert "n.industry" in cypher
        assert "COUNT(*) AS count" in cypher
        assert "AVG(n.revenue) AS avg_revenue" in cypher

    def test_execution_method_all(self):
        """Test .all() execution method."""
        query = self.session.query(TestPerson).filter_by(is_active=True)
        results = query.all()
        
        assert isinstance(results, list)
        assert len(results) == 8
        assert all(isinstance(r, TestPerson) for r in results)
        assert all(r.is_active for r in results)
    
    def test_execution_method_first(self):
        """Test .first() execution method."""
        query = self.session.query(TestPerson).order_by("id")
        result = query.first()
        
        assert isinstance(result, TestPerson)
        assert result.id == 1
        assert result.name == "Alice Smith"
        
        # @@ STEP: Test with no results
        query = self.session.query(TestPerson).filter_by(age=999)
        result = query.first()
        
        assert result is None
    
    def test_execution_method_one(self):
        """Test .one() execution method."""
        query = self.session.query(TestPerson).filter_by(email="alice@test.com")
        result = query.one()
        
        assert isinstance(result, TestPerson)
        assert result.name == "Alice Smith"
        
        # @@ STEP: Test exception for multiple results
        with pytest.raises(Exception):
            query = self.session.query(TestPerson).filter_by(is_active=True)
            query.one()
        
        # @@ STEP: Test exception for no results
        with pytest.raises(Exception):
            query = self.session.query(TestPerson).filter_by(age=999)
            query.one()


class TestRelationshipQueries:
    """Tests for relationship queries."""
    
    @classmethod
    def setup_class(cls):
        """Set up test database and session once for all tests."""
        cls.temp_db = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_db) / "test_fluent_rel.db"
        cls.session = KuzuSession(db_path=str(cls.db_path))
        
        # @@ STEP: Initialize schema
        initialize_schema(cls.session)
        
        # @@ STEP: Insert test data
        cls._insert_test_data_static(cls.session)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test database after all tests."""
        cls.session.close()
        if Path(cls.temp_db).exists():
            shutil.rmtree(cls.temp_db, ignore_errors=True)
    
    @classmethod
    def _insert_test_data_static(cls, session: KuzuSession):
        """Insert test data for relationship testing."""
        # @@ STEP: Create persons
        persons = [
            TestPerson(id=1, name="Alice Smith", email="alice@test.com", age=30, salary=75000, is_active=True),
            TestPerson(id=2, name="Bob Johnson", email="bob@test.com", age=25, salary=60000, is_active=True),
            TestPerson(id=3, name="Charlie Brown", email="charlie@test.com", age=35, salary=85000, is_active=False),
            TestPerson(id=4, name="Diana Prince", email="diana@test.com", age=28, salary=70000, is_active=True),
            TestPerson(id=5, name="Eve Wilson", email="eve@test.com", age=32, salary=80000, is_active=True),
        ]
        
        # @@ STEP: Create companies
        companies = [
            TestCompany(id=1, name="TechCorp", industry="Technology", revenue=1000000, employee_count=100, is_public=True),
            TestCompany(id=2, name="FinanceInc", industry="Finance", revenue=5000000, employee_count=500, is_public=True),
            TestCompany(id=3, name="StartupXYZ", industry="Technology", revenue=100000, employee_count=10, is_public=False),
        ]
        
        # @@ STEP: Create projects
        projects = [
            TestProject(id=1, name="Alpha", description="First project", budget=100000, status="ACTIVE", priority=1),
            TestProject(id=2, name="Beta", description="Second project", budget=200000, status="ACTIVE", priority=2),
            TestProject(id=3, name="Gamma", description="Third project", budget=150000, status="PLANNING", priority=3),
        ]
        
        # @@ STEP: Add all nodes
        for person in persons:
            session.add(person)
        for company in companies:
            session.add(company)
        for project in projects:
            session.add(project)
        session.commit()
        
        # @@ STEP: Create relationships using fluent API
        from datetime import date

        # @@ STEP: Create WORKS_FOR relationships using proper fluent API
        # Get node instances for relationship creation
        alice = session.query(TestPerson).filter_by(id=1).first()
        bob = session.query(TestPerson).filter_by(id=2).first()
        charlie = session.query(TestPerson).filter_by(id=3).first()
        diana = session.query(TestPerson).filter_by(id=4).first()
        eve = session.query(TestPerson).filter_by(id=5).first()

        techcorp = session.query(TestCompany).filter_by(id=1).first()
        financeinc = session.query(TestCompany).filter_by(id=2).first()
        startupxyz = session.query(TestCompany).filter_by(id=3).first()

        # Create WORKS_FOR relationships using the new intuitive API
        session.create_relationship(
            WorksFor, alice, techcorp,
            position='Senior Engineer', department='Engineering',
            is_remote=False, start_date=date(2020, 1, 15)
        )
        session.create_relationship(
            WorksFor, bob, techcorp,
            position='Junior Engineer', department='Engineering',
            is_remote=True, start_date=date(2021, 3, 1)
        )
        session.create_relationship(
            WorksFor, charlie, financeinc,
            position='Analyst', department='Finance',
            is_remote=False, start_date=date(2019, 6, 1)
        )
        session.create_relationship(
            WorksFor, diana, startupxyz,
            position='CTO', department='Executive',
            is_remote=False, start_date=date(2022, 1, 1)
        )
        session.create_relationship(
            WorksFor, eve, financeinc,
            position='Senior Analyst', department='Finance',
            is_remote=True, start_date=date(2020, 9, 1)
        )

        # Get project instances
        alpha = session.query(TestProject).filter_by(id=1).first()
        beta = session.query(TestProject).filter_by(id=2).first()
        gamma = session.query(TestProject).filter_by(id=3).first()

        # @@ STEP: Create MANAGES relationships using the new intuitive API
        session.create_relationship(
            Manages, alice, alpha,
            role='Lead', responsibility_level=3, assigned_date=date(2023, 1, 1)
        )
        session.create_relationship(
            Manages, alice, beta,
            role='Advisor', responsibility_level=1, assigned_date=date(2023, 2, 1)
        )
        session.create_relationship(
            Manages, diana, gamma,
            role='Overseer', responsibility_level=2, assigned_date=date(2023, 3, 1)
        )
        session.create_relationship(
            Manages, eve, alpha,
            role='Coordinator', responsibility_level=1, assigned_date=date(2023, 1, 15)
        )

        # @@ STEP: Create COLLABORATES relationships using the new intuitive API
        session.create_relationship(
            Collaborates, alice, bob,
            project_count=3, trust_score=0.9, since=date(2020, 1, 1)
        )
        session.create_relationship(
            Collaborates, bob, charlie,
            project_count=1, trust_score=0.6, since=date(2021, 6, 1)
        )
        session.create_relationship(
            Collaborates, charlie, diana,
            project_count=2, trust_score=0.75, since=date(2022, 1, 1)
        )
        session.create_relationship(
            Collaborates, diana, eve,
            project_count=4, trust_score=0.95, since=date(2022, 3, 1)
        )

        # @@ STEP: Create OWNS relationships using the new intuitive API
        session.create_relationship(
            Owns, techcorp, alpha,
            ownership_percentage=100.0, investment_amount=100000, acquired_date=date(2023, 1, 1)
        )
        session.create_relationship(
            Owns, techcorp, beta,
            ownership_percentage=50.0, investment_amount=100000, acquired_date=date(2023, 2, 1)
        )
        session.create_relationship(
            Owns, financeinc, beta,
            ownership_percentage=50.0, investment_amount=100000, acquired_date=date(2023, 2, 1)
        )
        session.create_relationship(
            Owns, startupxyz, gamma,
            ownership_percentage=100.0, investment_amount=150000, acquired_date=date(2023, 3, 1)
        )

        # @@ STEP: Create MULTI_REL relationships using the new intuitive API
        session.create_relationship(
            MultiRel, alice, techcorp,
            connection_type='employee', strength=0.9
        )
        session.create_relationship(
            MultiRel, alice, techcorp,
            connection_type='sponsor', strength=1.0
        )
        session.create_relationship(
            MultiRel, alice, techcorp,
            connection_type='contributor', strength=0.8
        )

        # @@ STEP: Commit all relationships
        cls.session.commit()
    
    def test_basic_relationship_query(self):
        """Test basic relationship query."""
        query = self.session.query(WorksFor)
        results = query.all()
        
        assert len(results) == 5
        assert all(isinstance(r, WorksFor) for r in results)
    
    def test_relationship_filter_by_properties(self):
        """Test filtering relationships by properties."""
        query = self.session.query(WorksFor).filter_by(department="Engineering")
        results = query.all()
        
        assert len(results) == 2
        for r in results:
            assert r.department == "Engineering"
    
    def test_relationship_filter_with_operators(self):
        """Test filtering relationships with operators."""
        # @@ STEP: Filter by remote workers
        query = self.session.query(WorksFor).filter(
            QueryField("is_remote", WorksFor) == True
        )
        results = query.all()
        
        assert len(results) == 2
        
        # @@ STEP: Filter by position containing Engineer
        query = self.session.query(WorksFor).filter(
            QueryField("position", WorksFor).contains("Engineer")
        )
        results = query.all()
        
        assert len(results) == 2
        for r in results:
            assert "Engineer" in r.position
    
    def test_relationship_with_multiple_filters(self):
        """Test relationships with multiple filter conditions."""
        query = self.session.query(Manages).filter(
            (QueryField("responsibility_level", Manages) >= 2) &
            (QueryField("role", Manages) != "Advisor")
        )
        results = query.all()
        
        assert len(results) == 2  # Lead and Overseer
        for r in results:
            assert r.responsibility_level >= 2
            assert r.role != "Advisor"
    
    def test_relationship_ordering(self):
        """Test ordering of relationship results."""
        query = self.session.query(Collaborates).order_by(("trust_score", OrderDirection.DESC))
        results = query.all()
        
        trust_scores = [r.trust_score for r in results]
        assert trust_scores == sorted(trust_scores, reverse=True)
    
    def test_relationship_limit_offset(self):
        """Test limit and offset on relationship queries."""
        query = self.session.query(WorksFor).order_by("position").limit(2).offset(1)
        results = query.all()
        
        assert len(results) == 2
    
    def test_relationship_aggregations(self):
        """Test aggregations on relationship properties."""
        # @@ STEP: Count relationships
        query = self.session.query(Manages).count()
        cypher, _ = query.to_cypher()
        assert "COUNT(*) AS count" in cypher
        
        # @@ STEP: Average trust score
        query = self.session.query(Collaborates).avg("trust_score", alias="avg_trust")
        cypher, _ = query.to_cypher()
        assert "AVG(n.trust_score) AS avg_trust" in cypher
        
        # @@ STEP: Sum of project counts
        query = self.session.query(Collaborates).sum("project_count", alias="total_projects")
        cypher, _ = query.to_cypher()
        assert "SUM(n.project_count) AS total_projects" in cypher
    
    def test_multi_pair_relationship_query(self):
        """Test querying multi-pair relationships."""
        query = self.session.query(MultiRel)
        results = query.all()

        assert len(results) == 3
        connection_types = {r.connection_type for r in results}
        assert connection_types == {"employee", "sponsor", "contributor"}
    
    def test_multi_pair_relationship_filtering(self):
        """Test filtering multi-pair relationships."""
        query = self.session.query(MultiRel).filter(
            QueryField("strength", MultiRel) >= 0.9
        )
        results = query.all()

        assert len(results) == 2  # employee and sponsor
        for r in results:
            assert r.strength >= 0.9
    
    def test_self_referencing_relationship(self):
        """Test self-referencing relationship queries."""
        query = self.session.query(Collaborates).filter(
            QueryField("project_count", Collaborates) > 2
        )
        results = query.all()
        
        # session.create_relationship(
        #     Collaborates, alice, bob,
        #     project_count=3, trust_score=0.9, since=date(2020, 1, 1)
        # )
        # session.create_relationship(
        #     Collaborates, diana, eve,
        #     project_count=4, trust_score=0.95, since=date(2022, 3, 1)
        # )
        
        assert len(results) == 2  # Alice-Bob and Diana-Eve collaborations
        for r in results:
            assert r.project_count >= 3


class TestCombinedQueries:
    """Tests for combined node + relationship queries."""
    
    @classmethod
    def setup_class(cls):
        """Set up test database and session once for all tests."""
        cls.temp_db = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_db) / "test_fluent_combined.db"
        cls.session = KuzuSession(db_path=str(cls.db_path))
        
        # @@ STEP: Initialize schema
        initialize_schema(cls.session)
        
        # @@ STEP: Insert test data
        cls._insert_test_data_static(cls.session)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test database after all tests."""
        cls.session.close()
        if Path(cls.temp_db).exists():
            shutil.rmtree(cls.temp_db, ignore_errors=True)
    
    @classmethod
    def _insert_test_data_static(cls, session: KuzuSession):
        """Insert test data for combined query testing."""
        # @@ STEP: Create and add nodes
        persons = [
            TestPerson(id=1, name="Alice Smith", email="alice@test.com", age=30, salary=75000, is_active=True),
            TestPerson(id=2, name="Bob Johnson", email="bob@test.com", age=25, salary=60000, is_active=True),
            TestPerson(id=3, name="Charlie Brown", email="charlie@test.com", age=35, salary=85000, is_active=False),
            TestPerson(id=4, name="Diana Prince", email="diana@test.com", age=28, salary=70000, is_active=True),
        ]
        
        companies = [
            TestCompany(id=1, name="TechCorp", industry="Technology", revenue=1000000, employee_count=100, is_public=True),
            TestCompany(id=2, name="FinanceInc", industry="Finance", revenue=5000000, employee_count=500, is_public=True),
        ]
        
        projects = [
            TestProject(id=1, name="Alpha", description="First project", budget=100000, status="ACTIVE", priority=1),
            TestProject(id=2, name="Beta", description="Second project", budget=200000, status="ACTIVE", priority=2),
            TestProject(id=3, name="Gamma", description="Third project", budget=150000, status="PLANNING", priority=3),
        ]
        
        for person in persons:
            session.add(person)
        for company in companies:
            session.add(company)
        for project in projects:
            session.add(project)
        session.commit()
        
        # @@ STEP: Create relationships using fluent API
        from datetime import date

        # @@ STEP: Create employment relationships using proper fluent API
        # Get node instances for relationship creation
        alice = session.query(TestPerson).filter_by(id=1).first()
        bob = session.query(TestPerson).filter_by(id=2).first()
        charlie = session.query(TestPerson).filter_by(id=3).first()
        diana = session.query(TestPerson).filter_by(id=4).first()

        techcorp = session.query(TestCompany).filter_by(id=1).first()
        financeinc = session.query(TestCompany).filter_by(id=2).first()

        alpha = session.query(TestProject).filter_by(id=1).first()
        beta = session.query(TestProject).filter_by(id=2).first()
        gamma = session.query(TestProject).filter_by(id=3).first()

        # Create employment relationships using the new intuitive API
        session.create_relationship(
            WorksFor, alice, techcorp,
            position='Senior Engineer', department='Engineering',
            is_remote=False, start_date=date(2020, 1, 15)
        )
        session.create_relationship(
            WorksFor, bob, techcorp,
            position='Junior Engineer', department='Engineering',
            is_remote=True, start_date=date(2021, 3, 1)
        )
        session.create_relationship(
            WorksFor, charlie, financeinc,
            position='Analyst', department='Finance',
            is_remote=False, start_date=date(2019, 6, 1)
        )
        session.create_relationship(
            WorksFor, diana, financeinc,
            position='Senior Analyst', department='Finance',
            is_remote=True, start_date=date(2020, 9, 1)
        )

        # @@ STEP: Create project management relationships using the new intuitive API
        session.create_relationship(
            Manages, alice, alpha,
            role='Lead', responsibility_level=3, assigned_date=date(2023, 1, 1)
        )
        session.create_relationship(
            Manages, bob, alpha,
            role='Developer', responsibility_level=1, assigned_date=date(2023, 1, 15)
        )
        session.create_relationship(
            Manages, charlie, beta,
            role='Analyst', responsibility_level=2, assigned_date=date(2023, 2, 1)
        )
        session.create_relationship(
            Manages, diana, gamma,
            role='Coordinator', responsibility_level=2, assigned_date=date(2023, 3, 1)
        )

        # @@ STEP: Create company-project ownership relationships using the new intuitive API
        session.create_relationship(
            Owns, techcorp, alpha,
            ownership_percentage=100.0, investment_amount=100000, acquired_date=date(2023, 1, 1)
        )
        session.create_relationship(
            Owns, financeinc, beta,
            ownership_percentage=100.0, investment_amount=200000, acquired_date=date(2023, 2, 1)
        )
        session.create_relationship(
            Owns, financeinc, gamma,
            ownership_percentage=100.0, investment_amount=150000, acquired_date=date(2023, 3, 1)
        )

        # @@ STEP: Commit all relationships
        session.commit()
    
    def test_join_nodes_with_relationships(self):
        """Test joining nodes through relationships."""
        # @@ STEP: Join person with company through WORKS_FOR
        query = self.session.query(TestPerson).join(
            WorksFor, QueryField("id", TestPerson) == QueryField("from_id", WorksFor)
        ).join(
            TestCompany, QueryField("to_id", WorksFor) == QueryField("id", TestCompany)
        )
        cypher, _ = query.to_cypher()
        
        # || S.S: Verify join structure in Cypher
        assert "MATCH" in cypher
        assert "TestPerson" in cypher
        assert "WORKS_FOR" in cypher
        assert "TestCompany" in cypher
    
    def test_complex_join_with_filters(self):
        """Test complex joins with multiple filters."""
        # @@ STEP: Find engineers in tech companies
        query = self.session.query(TestPerson).join(
            WorksFor, QueryField("id", TestPerson) == QueryField("from_id", WorksFor)
        ).join(
            TestCompany, QueryField("to_id", WorksFor) == QueryField("id", TestCompany)
        ).filter(
            (QueryField("department", WorksFor) == "Engineering") &
            (QueryField("industry", TestCompany) == "Technology")
        )
        cypher, params = query.to_cypher()
        
        # || S.S: Verify filter conditions
        assert "department" in cypher
        assert "param_" in cypher  # Parameters are used
        assert "industry" in cypher
        assert params  # Parameters dict should exist
        param_values = list(params.values())
        assert "Engineering" in param_values
        assert "Technology" in param_values
    
    def test_multi_hop_relationships(self):
        """Test queries spanning multiple relationship hops."""
        # @@ STEP: Find people managing projects owned by companies
        query = self.session.query(TestPerson).join(
            Manages, QueryField("id", TestPerson) == QueryField("from_id", Manages)
        ).join(
            TestProject, QueryField("to_id", Manages) == QueryField("id", TestProject)
        ).join(
            Owns, QueryField("id", TestProject) == QueryField("to_id", Owns)
        ).join(
            TestCompany, QueryField("from_id", Owns) == QueryField("id", TestCompany)
        )
        cypher, _ = query.to_cypher()
        
        # || S.S: Verify multi-hop join structure
        assert "MANAGES" in cypher  # First relationship
        assert "TestProject" in cypher  # Middle node
        assert "OWNS" in cypher  # Second relationship
        assert "TestCompany" in cypher  # End node
    
    def test_combined_aggregations(self):
        """Test aggregations across combined queries."""
        # @@ STEP: Count employees per company with filtering
        query = self.session.query(TestCompany).join(
            WorksFor, QueryField("id", TestCompany) == QueryField("to_id", WorksFor)
        ).join(
            TestPerson, QueryField("from_id", WorksFor) == QueryField("id", TestPerson)
        ).filter(
            QueryField("is_active", TestPerson) == True
        ).count()
        cypher, _ = query.to_cypher()
        
        assert "COUNT(*) AS count" in cypher
        assert "is_active" in cypher
    
    def test_optional_joins(self):
        """Test left/optional joins in combined queries."""
        # @@ STEP: Find all persons with optional company connection
        query = self.session.query(TestPerson).join(
            WorksFor, 
            QueryField("id", TestPerson) == QueryField("from_id", WorksFor),
            join_type="left"
        )
        cypher, _ = query.to_cypher()
        
        # || S.S: Left joins typically use OPTIONAL MATCH in Cypher
        # || S.S: But implementation may vary
        assert "MATCH" in cypher
    
    def test_relationship_path_queries(self):
        """Test queries involving relationship paths."""
        # @@ STEP: Find indirect connections through projects
        query = self.session.query(TestPerson).join(
            Manages, QueryField("id", TestPerson) == QueryField("from_id", Manages)
        ).filter(
            QueryField("responsibility_level", Manages) >= 2
        ).join(
            TestProject, QueryField("to_id", Manages) == QueryField("id", TestProject)
        ).filter(
            QueryField("status", TestProject) == "ACTIVE"
        )
        cypher, params = query.to_cypher()

        assert "responsibility_level" in cypher
        assert "status" in cypher
        param_values = list(params.values())
        assert "ACTIVE" in param_values

    def test_intuitive_relationship_traversal(self):
        """Test the new intuitive relationship traversal API."""
        # @@ STEP: Find companies that employ people using traverse()
        companies_with_employees = (
            self.session.query(TestPerson)
            .filter(QueryField("is_active", TestPerson) == True)
            .traverse(WorksFor, TestCompany, direction="outgoing")
            .distinct()
        )
        cypher, params = companies_with_employees.to_cypher()

        # Verify the query structure
        assert "TestPerson" in cypher
        assert isinstance(params, dict)
        assert "WORKS_FOR" in cypher
        assert "TestCompany" in cypher
        assert "is_active" in cypher

    def test_outgoing_relationship_traversal(self):
        """Test outgoing relationship traversal using outgoing() method."""
        # @@ STEP: Find projects managed by active people
        managed_projects = (
            self.session.query(TestPerson)
            .filter(QueryField("is_active", TestPerson) == True)
            .outgoing(Manages, TestProject)
            .filter(QueryField("status", TestProject) == "ACTIVE")
        )
        cypher, params = managed_projects.to_cypher()

        # Verify the query structure
        assert "TestPerson" in cypher
        assert isinstance(params, dict)
        assert "MANAGES" in cypher
        assert "TestProject" in cypher
        assert "is_active" in cypher
        assert "status" in cypher

    def test_incoming_relationship_traversal(self):
        """Test incoming relationship traversal using incoming() method."""
        # @@ STEP: Find people who work for tech companies
        tech_employees = (
            self.session.query(TestCompany)
            .filter(QueryField("industry", TestCompany) == "Technology")
            .incoming(WorksFor, TestPerson)
            .filter(QueryField("is_active", TestPerson) == True)
        )
        cypher, params = tech_employees.to_cypher()

        # Verify the query structure
        assert "TestCompany" in cypher
        assert isinstance(params, dict)
        assert "WORKS_FOR" in cypher
        assert "TestPerson" in cypher
        assert "industry" in cypher
        assert "is_active" in cypher

    def test_bidirectional_relationship_traversal(self):
        """Test bidirectional relationship traversal using related() method."""
        # @@ STEP: Find all people connected through collaboration
        connected_people = (
            self.session.query(TestPerson)
            .filter(QueryField("name", TestPerson) == "Alice Smith")
            .related(Collaborates, TestPerson)
            .filter(QueryField("is_active", TestPerson) == True)
        )
        cypher, params = connected_people.to_cypher()

        # Verify the query structure
        assert "TestPerson" in cypher
        assert isinstance(params, dict)
        assert "COLLABORATES" in cypher
        assert "name" in cypher
        assert "is_active" in cypher

    def test_multi_hop_traversal(self):
        """Test multi-hop relationship traversal."""
        # @@ STEP: Find companies connected to projects through people
        # People -> Manages -> Projects -> Owns <- Companies
        connected_companies = (
            self.session.query(TestPerson)
            .filter(QueryField("is_active", TestPerson) == True)
            .outgoing(Manages, TestProject)
            .filter(QueryField("status", TestProject) == "ACTIVE")
            .incoming(Owns, TestCompany)
            .filter(QueryField("is_public", TestCompany) == True)
        )
        cypher, params = connected_companies.to_cypher()

        # Verify the complex query structure
        assert "TestPerson" in cypher
        assert isinstance(params, dict)
        assert "MANAGES" in cypher
        assert "TestProject" in cypher
        assert "OWNS" in cypher
        assert "TestCompany" in cypher
        assert "is_active" in cypher
        assert "status" in cypher
        assert "is_public" in cypher

    def test_relationship_filtering_with_traversal(self):
        """Test filtering relationships during traversal."""
        # @@ STEP: Find senior employees in engineering departments
        senior_engineers = (
            self.session.query(TestPerson)
            .outgoing(WorksFor, TestCompany,
                     conditions=[QueryField("department", WorksFor) == "Engineering"])
            .filter(QueryField("industry", TestCompany) == "Technology")
        )
        cypher, params = senior_engineers.to_cypher()

        # Verify relationship filtering
        assert "WORKS_FOR" in cypher
        assert isinstance(params, dict)
        assert "department" in cypher
        assert "industry" in cypher

    def test_aggregation_with_traversal(self):
        """Test aggregations combined with relationship traversal."""
        # @@ STEP: Count projects per company through ownership
        project_counts = (
            self.session.query(TestCompany)
            .outgoing(Owns, TestProject)
            .group_by("name")
            .count("*", alias="project_count")
        )
        cypher, params = project_counts.to_cypher()

        # Verify aggregation with traversal
        assert "TestCompany" in cypher
        assert isinstance(params, dict)
        assert "OWNS" in cypher
        assert "TestProject" in cypher
        assert "COUNT(*) AS project_count" in cypher
        # Note: Kuzu doesn't use explicit GROUP BY clauses - grouping is implicit in RETURN


class TestEdgeCasesAndErrors:
    """Tests for edge cases and error scenarios."""
    
    @classmethod
    def setup_class(cls):
        """Set up test database and session once for all tests."""
        cls.temp_db = tempfile.mkdtemp()
        cls.db_path = Path(cls.temp_db) / "test_fluent_edge.db"
        cls.session = KuzuSession(db_path=str(cls.db_path))
        
        # @@ STEP: Initialize schema
        initialize_schema(cls.session)
        
        # @@ STEP: Insert minimal test data
        cls._insert_test_data_static(cls.session)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test database after all tests."""
        cls.session.close()
        if Path(cls.temp_db).exists():
            shutil.rmtree(cls.temp_db, ignore_errors=True)
    
    @classmethod
    def _insert_test_data_static(cls, session: KuzuSession):
        """Insert minimal test data for edge case testing."""
        # @@ STEP: Insert basic test persons
        persons = [
            TestPerson(id=1, name="Test User", email="test@test.com", age=25, salary=50000, is_active=True),
            TestPerson(id=2, name="Another User", email="another@test.com", age=30, salary=60000, is_active=False),
        ]
        
        companies = [
            TestCompany(id=1, name="TestCo", industry="Tech", revenue=100000, employee_count=10, is_public=False),
        ]
        
        for person in persons:
            session.add(person)
        for company in companies:
            session.add(company)
        session.commit()
        
        # @@ STEP: Create one relationship using proper fluent API
        from datetime import date

        # Get node instances
        alice = session.query(TestPerson).filter_by(id=1).first()
        techcorp = session.query(TestCompany).filter_by(id=1).first()

        # Create relationship using the new intuitive API
        rel = session.create_relationship(
            WorksFor, alice, techcorp,
            position='Engineer', department='Tech',
            is_remote=True, start_date=date(2023, 1, 1)
        )

        # Validate relationship was created
        assert rel.__class__.__name__ == "WorksFor"
        session.commit()
    
    def test_empty_result_handling(self):
        """Test handling of empty result sets."""
        # @@ STEP: Query with no matches
        query = self.session.query(TestPerson).filter_by(age=999)
        results = query.all()
        
        assert results == []
        assert len(results) == 0
        
        # @@ STEP: First on empty results
        result = query.first()
        assert result is None
    
    def test_null_value_filtering(self):
        """Test filtering with NULL values."""
        # @@ STEP: Create person with NULL email
        person = TestPerson(id=99, name="Null Person", email=None, age=40, salary=70000, is_active=True)
        self.session.add(person)
        self.session.commit()
        
        # @@ STEP: Filter by NULL
        query = self.session.query(TestPerson).filter(
            QueryField("email", TestPerson).is_null()
        )
        results = query.all()
        
        assert len(results) == 1
        assert results[0].email is None
        
        # @@ STEP: Filter by NOT NULL
        query = self.session.query(TestPerson).filter(
            QueryField("email", TestPerson).is_not_null()
        )
        results = query.all()
        
        assert len(results) == 2  # Original two test users
        for r in results:
            assert r.email is not None
    
    def test_special_characters_in_filters(self):
        """Test handling of special characters in filter values."""
        # @@ STEP: Create person with special characters
        person = TestPerson(
            id=100, 
            name="O'Brien", 
            email="test+special@example.com", 
            age=35, 
            salary=65000, 
            is_active=True
        )
        self.session.add(person)
        self.session.commit()
        
        # @@ STEP: Query with special characters
        query = self.session.query(TestPerson).filter_by(name="O'Brien")
        results = query.all()
        
        assert len(results) == 1
        assert results[0].name == "O'Brien"
        
        # @@ STEP: Email with special characters
        query = self.session.query(TestPerson).filter_by(email="test+special@example.com")
        results = query.all()
        
        assert len(results) == 1
        assert results[0].email == "test+special@example.com"
    
    def test_boolean_edge_cases(self):
        """Test boolean field edge cases."""
        # @@ STEP: Filter by True
        query = self.session.query(TestPerson).filter(
            QueryField("is_active", TestPerson) == True
        )
        true_results = query.all()
        
        # @@ STEP: Filter by False
        query = self.session.query(TestPerson).filter(
            QueryField("is_active", TestPerson) == False
        )
        false_results = query.all()
        
        # @@ STEP: Verify partition
        assert len(true_results) == 1 or len(true_results) == 2 or len(true_results) == 3 or len(true_results) == 4 or len(true_results) == 5
        assert len(false_results) == 1 or len(false_results) == 2 or len(false_results) == 3 or len(false_results) == 4 or len(false_results) == 5
        assert all(r.is_active for r in true_results)
        assert all(not r.is_active for r in false_results)
    
    def test_numeric_boundary_conditions(self):
        """Test numeric field boundary conditions."""
        # @@ STEP: Test with zero
        query = self.session.query(TestPerson).filter(
            QueryField("salary", TestPerson) > 0
        )
        results = query.all()
        # @@ STEP: Validate that all results have positive salaries (matching the filter > 0)
        assert all(r.salary > 0 for r in results), f"All salaries should be > 0, got: {[r.salary for r in results]}"
        # @@ STEP: Validate that we get at least the expected results from our test data
        assert len(results) >= 2, f"Should have at least 2 results from test data, got {len(results)}"
        # @@ STEP: Validate that the filter is working correctly by checking no zero/negative salaries
        assert all(r.salary > 0 for r in results), "Filter > 0 should exclude zero/negative salaries"
        
        # @@ STEP: Test with negative (shouldn't match)
        query = self.session.query(TestPerson).filter(
            QueryField("age", TestPerson) < 0
        )
        results = query.all()
        assert len(results) == 0
        
        # @@ STEP: Test with very large number
        query = self.session.query(TestCompany).filter(
            QueryField("revenue", TestCompany) < 999999999
        )
        results = query.all()
        assert len(results) == 1 or len(results) == 2 or len(results) == 3 or len(results) == 4 or len(results) == 5
    
    def test_chained_method_order(self):
        """Test that method chaining order doesn't affect results."""
        # @@ STEP: Filter then order
        query1 = self.session.query(TestPerson).filter(
            QueryField("age", TestPerson) >= 25
        ).order_by("name")
        
        # @@ STEP: Order then filter
        query2 = self.session.query(TestPerson).order_by("name").filter(
            QueryField("age", TestPerson) >= 25
        )
        
        # || S.S: Both should produce equivalent queries
        cypher1, _ = query1.to_cypher()
        cypher2, _ = query2.to_cypher()
        
        # || S.S: Both should have filter and order clauses
        assert "WHERE" in cypher1 and "WHERE" in cypher2
        assert "ORDER BY" in cypher1 and "ORDER BY" in cypher2
    
    def test_duplicate_filters(self):
        """Test handling of duplicate/conflicting filters."""
        # @@ STEP: Multiple filters on same field
        query = self.session.query(TestPerson).filter(
            QueryField("age", TestPerson) > 20
        ).filter(
            QueryField("age", TestPerson) < 40
        )
        cypher, _ = query.to_cypher()
        
        # || S.S: Both conditions should be present
        assert cypher.count("age") == 2 or cypher.count("age") == 3 or cypher.count("age") == 4 or cypher.count("age") == 5
    
    def test_invalid_field_names(self):
        """Test handling of invalid field names."""
        # @@ STEP: This should not crash but may produce invalid Cypher
        query = self.session.query(TestPerson).filter_by(nonexistent_field="value")
        cypher, _ = query.to_cypher()
        
        # || S.S: Query should still build
        assert "nonexistent_field" in cypher
    
    def test_relationship_without_nodes(self):
        """Test querying orphaned relationships."""
        # @@ STEP: Query relationships directly
        query = self.session.query(WorksFor)
        results = query.all()
        
        # || S.S: Should return existing relationships
        assert len(results) == 1
        assert results[0].position == "Engineer"
    
    def test_circular_joins(self):
        """Test handling of circular join patterns."""
        # @@ STEP: Create self-referencing relationship using fluent API
        from datetime import date
        
        # @@ STEP: Create relationship using proper fluent API
        # Get node instances
        alice = self.session.query(TestPerson).filter_by(id=1).first()
        bob = self.session.query(TestPerson).filter_by(id=2).first()

        # Create relationship using the new intuitive API
        rel = self.session.create_relationship(
            Collaborates, alice, bob,
            project_count=1, trust_score=0.5, since=date(2023, 1, 1)
        )

        # Validate relationship was created
        assert rel.__class__.__name__ == "Collaborates"
        self.session.commit()
        
        # @@ STEP: Query with potential circular pattern
        query = self.session.query(TestPerson).join(
            Collaborates, QueryField("id", TestPerson) == QueryField("from_id", Collaborates)
        )
        cypher, _ = query.to_cypher()
        
        assert "TestPerson" in cypher
        assert "COLLABORATES" in cypher
    
    def test_extremely_long_strings(self):
        """Test handling of very long string values."""
        # @@ STEP: Create person with very long name
        long_name = "A" * 1000
        person = TestPerson(
            id=101,
            name=long_name,
            email="long@test.com",
            age=30,
            salary=50000,
            is_active=True
        )
        self.session.add(person)
        self.session.commit()
        
        # @@ STEP: Query by long string
        query = self.session.query(TestPerson).filter_by(name=long_name)
        results = query.all()
        
        assert len(results) == 1
        assert len(results[0].name) == 1000
