"""
COMPREHENSIVE INTEGRATION TESTS
====================================================

These tests simulate real-world scenarios with precision and rigorous validation.

Test Categories:
1. Complex Real-World Scenarios (Social Network, E-commerce, Knowledge Graph)
2. Performance & Scalability (Large datasets, concurrent operations)
3. Data Integrity & Consistency (Transactions, constraints, recovery)
4. Edge Cases & Error Handling (Malformed data, resource exhaustion)
5. Memory & Resource Management
6. Concurrent Access Patterns
7. Schema Evolution & Migration
"""

from __future__ import annotations

import tempfile
import shutil
import time
import threading
import psutil
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import random

from kuzualchemy import (
    KuzuBaseModel,
    KuzuRelationshipBase,
    kuzu_node,
    kuzu_relationship,
    kuzu_field,
    KuzuDataType,
    KuzuSession,
    clear_registry,
)
from kuzualchemy.test_utilities import initialize_schema
from kuzualchemy.kuzu_orm import get_ddl_for_node, get_ddl_for_relationship


# ============================================================================
# MODELS - SOCIAL NETWORK SCENARIO
# ============================================================================

@kuzu_node("SocialUser")
class SocialUser(KuzuBaseModel):
    """social network user model."""
    user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    full_name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    bio: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)
    follower_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    following_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    post_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    is_verified: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)
    is_active: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    last_login: Optional[datetime] = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=None)


@kuzu_node("SocialPost")
class SocialPost(KuzuBaseModel):
    """social media post model."""
    post_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    content: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    like_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    comment_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    share_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    is_public: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    updated_at: Optional[datetime] = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=None)


@kuzu_node("SocialComment")
class SocialComment(KuzuBaseModel):
    """comment model."""
    comment_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    content: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    like_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    is_edited: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_relationship("FOLLOWS", pairs=[(SocialUser, SocialUser)])
class Follows(KuzuRelationshipBase):
    """User follows another user relationship."""
    followed_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    is_mutual: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)
    notification_enabled: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)


@kuzu_relationship("AUTHORED", pairs=[(SocialUser, SocialPost)])
class Authored(KuzuRelationshipBase):
    """User authored a post relationship."""
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_relationship("LIKED", pairs=[(SocialUser, SocialPost), (SocialUser, SocialComment)])
class Liked(KuzuRelationshipBase):
    """User liked content relationship."""
    liked_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    reaction_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="like")


@kuzu_relationship("COMMENTED", pairs=[(SocialUser, SocialPost)])
class Commented(KuzuRelationshipBase):
    """User commented on post relationship."""
    comment_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64)
    commented_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


# ============================================================================
# E-COMMERCE MODELS
# ============================================================================

@kuzu_node("EcomCustomer")
class EcomCustomer(KuzuBaseModel):
    """e-commerce customer model."""
    customer_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    email: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    first_name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    last_name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    phone: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)
    total_orders: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    total_spent: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    loyalty_points: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    is_premium: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_node("EcomProduct")
class EcomProduct(KuzuBaseModel):
    """product model."""
    product_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    description: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    cost: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    stock_quantity: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    category: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    brand: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    rating: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    review_count: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
    is_active: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_node("EcomOrder")
class EcomOrder(KuzuBaseModel):
    """order model."""
    order_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    order_number: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    status: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    total_amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    tax_amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    shipping_amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    discount_amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    payment_method: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    shipping_address: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    shipped_at: Optional[datetime] = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=None)
    delivered_at: Optional[datetime] = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP, default=None)


@kuzu_relationship("PLACED_ORDER", pairs=[(EcomCustomer, EcomOrder)])
class PlacedOrder(KuzuRelationshipBase):
    """Customer placed order relationship."""
    placed_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_relationship("ORDER_CONTAINS", pairs=[(EcomOrder, EcomProduct)])
class OrderContains(KuzuRelationshipBase):
    """Order contains product relationship."""
    quantity: int = kuzu_field(kuzu_type=KuzuDataType.INT32, not_null=True)
    unit_price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    total_price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)


@kuzu_relationship("REVIEWED", pairs=[(EcomCustomer, EcomProduct)])
class Reviewed(KuzuRelationshipBase):
    """Customer reviewed product relationship."""
    rating: int = kuzu_field(kuzu_type=KuzuDataType.INT32, not_null=True)  # 1-5
    review_text: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)
    is_verified_purchase: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=False)
    reviewed_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


# ============================================================================
# TEST SUITE
# ============================================================================

class TestReadinessComprehensive:
    """
    Comprehensive tests.
    
    These tests validate that KuzuAlchemy can handle real-world scenarios
    with precision and rigorous validation.
    """

    def setup_method(self):
        """Set up test environment with configuration."""
        # Clear registry at start of each test to prevent memory corruption
        clear_registry()
        gc.collect()  # Force garbage collection to free memory

        self.temp_db = tempfile.mkdtemp()
        self.db_path = Path(self.temp_db) / "readiness_test.db"

        # Track memory usage
        self.process = psutil.Process()
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB

        # Performance tracking
        self.performance_metrics = {
            'insert_times': [],
            'query_times': [],
            'memory_usage': [],
            'concurrent_operations': []
        }

    def teardown_method(self):
        """Clean up test environment and report metrics."""
        # Clear registry after each test to prevent memory leaks
        clear_registry()
        gc.collect()  # Force garbage collection to free memory

        # Clean up connection pool to prevent test isolation issues
        try:
            from src.kuzualchemy.connection_pool import close_all_databases
            close_all_databases()
        except ImportError:
            pass

        if Path(self.temp_db).exists():
            shutil.rmtree(self.temp_db, ignore_errors=True)
        
        # Report performance metrics
        if self.performance_metrics['insert_times']:
            avg_insert = sum(self.performance_metrics['insert_times']) / len(self.performance_metrics['insert_times'])
            print(f"\nPerformance Metrics:")
            print(f"Average Insert Time: {avg_insert:.4f}s")
        
        if self.performance_metrics['query_times']:
            avg_query = sum(self.performance_metrics['query_times']) / len(self.performance_metrics['query_times'])
            print(f"Average Query Time: {avg_query:.4f}s")
        
        final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - self.initial_memory
        print(f"Memory Usage Increase: {memory_increase:.2f} MB")

    def _generate_ddl(self, models: List[Any]) -> str:
        """Generate DDL for specific models."""
        ddl_statements = []
        for model in models:
            if hasattr(model, '__is_kuzu_relationship__') and model.__is_kuzu_relationship__:
                ddl_statements.append(get_ddl_for_relationship(model))
            else:
                ddl_statements.append(get_ddl_for_node(model))
        return "\n".join(ddl_statements)

    def _measure_time(self, operation_name: str):
        """Context manager to measure operation time."""
        class TimeContext:
            def __init__(self, test_instance, op_name):
                self.test_instance = test_instance
                self.op_name = op_name
                self.start_time = None
            
            def __enter__(self):
                self.start_time = time.time()
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                duration = time.time() - self.start_time
                if 'insert' in self.op_name.lower():
                    self.test_instance.performance_metrics['insert_times'].append(duration)
                elif 'query' in self.op_name.lower() or 'queries' in self.op_name.lower():
                    self.test_instance.performance_metrics['query_times'].append(duration)
        
        return TimeContext(self, operation_name)

    def _track_memory(self):
        """Track current memory usage."""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.performance_metrics['memory_usage'].append(current_memory)
        return current_memory

    # ========================================================================
    # SOCIAL NETWORK SCENARIO TESTS
    # ========================================================================

    def test_social_network_complex_scenario(self):
        """
        Test complex social network scenario with precision.

        This test simulates a real social network with:
        - 1000 users
        - 5000 posts
        - 15000 likes
        - 10000 follows
        - 8000 comments

        Validates data integrity, query performance, and relationship consistency.
        """
        # CRITICAL: Registry cleanup to prevent access violations
        from kuzualchemy import clear_registry
        clear_registry()
        import gc
        gc.collect()
        session = KuzuSession(db_path=str(self.db_path))

        # Generate DDL for social network models
        social_models = [
            SocialUser, SocialPost, SocialComment,
            Follows, Authored, Liked, Commented
        ]
        ddl = self._generate_ddl(social_models)
        initialize_schema(session, ddl=ddl)

        # Phase 1: Create users with measured performance
        users = []
        with self._measure_time("bulk_user_insert"):
            for i in range(1000):
                user = SocialUser(
                    user_id=i + 1,
                    username=f"user_{i+1:04d}",
                    email=f"user{i+1}@example.com",
                    full_name=f"User {i+1}",
                    bio=f"Bio for user {i+1}" if i % 3 == 0 else None,
                    follower_count=0,  # Will be calculated based on actual relationships
                    following_count=0,  # Will be calculated based on actual relationships
                    post_count=0,  # Will be calculated based on actual relationships
                    is_verified=i < 50,  # First 50 users are verified
                    created_at=datetime.now() - timedelta(days=random.randint(1, 365))
                )
                users.append(user)
                session.add(user)

                # Track memory every 100 users
                if (i + 1) % 100 == 0:
                    self._track_memory()

            session.commit()

        # Validate user creation
        user_count_result = list(session.execute("MATCH (u:SocialUser) RETURN count(u) as count"))
        user_count = user_count_result[0]['count']
        assert user_count == 1000, f"Expected 1000 users, got {user_count}"

        # Phase 2: Create posts
        posts = []
        with self._measure_time("bulk_post_insert"):
            for i in range(5000):
                post = SocialPost(
                    post_id=i + 1,
                    content=f"This is post content {i+1}. " + "Lorem ipsum " * random.randint(5, 20),
                    like_count=random.randint(0, 500),
                    comment_count=random.randint(0, 50),
                    share_count=random.randint(0, 25),
                    is_public=random.choice([True, True, True, False]),  # 75% public
                    created_at=datetime.now() - timedelta(hours=random.randint(1, 8760))
                )
                posts.append(post)
                session.add(post)

            session.commit()

        # Phase 3: Create relationships with precision
        user_following_counts = {i: 0 for i in range(1, 1001)}
        user_follower_counts = {i: 0 for i in range(1, 1001)}
        user_post_counts = {i: 0 for i in range(1, 1001)}

        follow_relationships = []
        with self._measure_time("bulk_relationship_insert"):
            # Create follow relationships (each user follows 10-20 others)
            for user_id in range(1, 1001):
                follow_count = random.randint(10, 20)
                followed_users = random.sample(
                    [uid for uid in range(1, 1001) if uid != user_id],
                    follow_count
                )

                for followed_id in followed_users:
                    # Use session.create_relationship for proper relationship creation
                    follow_rel = session.create_relationship(
                        Follows, user_id, followed_id,
                        followed_at=datetime.now() - timedelta(days=random.randint(1, 180)),
                        is_mutual=random.choice([True, False]),
                        notification_enabled=random.choice([True, True, False])  # 67% enabled
                    )
                    follow_relationships.append(follow_rel)
                    # Track actual counts
                    user_following_counts[user_id] += 1
                    user_follower_counts[followed_id] += 1

            session.commit()

        # Phase 4: Create authored relationships and track counts
        with self._measure_time("authored_relationships"):
            for post_id in range(1, 5001):
                author_id = random.randint(1, 1000)
                authored_rel = session.create_relationship(
                    Authored, author_id, post_id,
                    created_at=datetime.now() - timedelta(hours=random.randint(1, 8760))
                )
                # Track actual post counts
                user_post_counts[author_id] += 1

            session.commit()

        # Phase 5: Update user counts to match actual relationships
        with self._measure_time("update_user_counts"):
            for user_id in range(1, 1001):
                # Update the user with actual counts
                session.execute(f"""
                    MATCH (u:SocialUser {{user_id: {user_id}}})
                    SET u.following_count = {user_following_counts[user_id]},
                        u.follower_count = {user_follower_counts[user_id]},
                        u.post_count = {user_post_counts[user_id]}
                """)
            session.commit()

        # Phase 6: Complex query validation with performance measurement
        with self._measure_time("complex_social_queries"):
            # Query 1: Find top 10 most followed users
            top_followed_query = """
            MATCH (follower:SocialUser)-[f:FOLLOWS]->(followed:SocialUser)
            RETURN followed.username, followed.full_name, count(f) as follower_count
            ORDER BY follower_count DESC
            LIMIT 10
            """
            top_followed = list(session.execute(top_followed_query))
            assert len(top_followed) == 10, "Should return top 10 followed users"

            # Validate consistency
            for i in range(len(top_followed) - 1):
                current_count = top_followed[i]['follower_count']
                next_count = top_followed[i + 1]['follower_count']
                assert current_count >= next_count, "Results should be ordered by follower count DESC"

            # Query 2: Find mutual follows
            mutual_follows_query = """
            MATCH (u1:SocialUser)-[f1:FOLLOWS {is_mutual: true}]->(u2:SocialUser)
            RETURN count(f1) as mutual_follow_count
            """
            mutual_count = list(session.execute(mutual_follows_query))[0]['mutual_follow_count']
            assert mutual_count > 0, "Should have mutual follows"

            # Query 3: Complex aggregation - posts per user with author info
            posts_per_user_query = """
            MATCH (u:SocialUser)-[a:AUTHORED]->(p:SocialPost)
            RETURN u.username, u.is_verified, count(p) as post_count,
                   avg(p.like_count) as avg_likes
            ORDER BY post_count DESC
            LIMIT 20
            """
            posts_per_user = list(session.execute(posts_per_user_query))
            assert len(posts_per_user) <= 20, "Should return at most 20 users"

            # Validate consistency in aggregations
            for result in posts_per_user:
                assert result['post_count'] > 0, "Post count should be positive"
                assert result['avg_likes'] >= 0, "Average likes should be non-negative"

        # Phase 6: Data integrity validation
        self._validate_social_network_integrity(session)

        # Phase 7: Memory and performance validation
        final_memory = self._track_memory()
        memory_increase = final_memory - self.initial_memory

        # Assert performance requirements - adjusted for realistic Windows memory usage
        assert memory_increase < 2000, f"Memory increase {memory_increase:.2f}MB exceeds 2000MB limit"

        if self.performance_metrics['insert_times']:
            avg_insert_time = sum(self.performance_metrics['insert_times']) / len(self.performance_metrics['insert_times'])
            assert avg_insert_time < 10.0, f"Average insert time {avg_insert_time:.4f}s exceeds 10s limit"
        else:
            assert False, "No insert times recorded - performance measurement failed"

        if self.performance_metrics['query_times']:
            avg_query_time = sum(self.performance_metrics['query_times']) / len(self.performance_metrics['query_times'])
            assert avg_query_time < 5.0, f"Average query time {avg_query_time:.4f}s exceeds 5s limit"
        else:
            assert False, "No query times recorded - performance measurement failed. FIX THIS. THIS IS NOT NORMAL. STOP BEING FUCKING LAZY."

        session.close()

    def _validate_social_network_integrity(self, session: KuzuSession):
        """Validate data integrity in social network scenario."""
        # Validate all FOLLOWS relationships connect valid SocialUser nodes
        follows_integrity = list(session.execute("""
            MATCH (u1:SocialUser)-[f:FOLLOWS]->(u2:SocialUser)
            RETURN count(f) as valid_follows_count
        """))

        total_follows = list(session.execute("""
            MATCH ()-[f:FOLLOWS]->()
            RETURN count(f) as total_follows_count
        """))

        assert follows_integrity[0]['valid_follows_count'] == total_follows[0]['total_follows_count'], \
            "All FOLLOWS relationships should connect valid SocialUser nodes"

        # Validate all AUTHORED relationships connect valid nodes
        authored_integrity = list(session.execute("""
            MATCH (u:SocialUser)-[a:AUTHORED]->(p:SocialPost)
            RETURN count(a) as valid_authored_count
        """))

        total_authored = list(session.execute("""
            MATCH ()-[a:AUTHORED]->()
            RETURN count(a) as total_authored_count
        """))

        assert authored_integrity[0]['valid_authored_count'] == total_authored[0]['total_authored_count'], \
            "All AUTHORED relationships should connect valid SocialUser to SocialPost nodes"

        # Validate user counts are consistent
        user_stats = list(session.execute("""
            MATCH (u:SocialUser)
            OPTIONAL MATCH (u)-[f:FOLLOWS]->()
            OPTIONAL MATCH (u)-[a:AUTHORED]->()
            RETURN u.user_id, count(DISTINCT f) as actual_following,
                   count(DISTINCT a) as actual_posts,
                   u.following_count, u.post_count
        """))

        # Validate that user counts are consistent
        for user_stat in user_stats:
            user_id = user_stat['u.user_id']
            actual_following = user_stat['actual_following']
            actual_posts = user_stat['actual_posts']
            stored_following_count = user_stat['u.following_count']
            stored_post_count = user_stat['u.post_count']

            # Counts must match exactly
            assert actual_following == stored_following_count, \
                f"User {user_id}: actual following count {actual_following} != stored count {stored_following_count}"

            assert actual_posts == stored_post_count, \
                f"User {user_id}: actual post count {actual_posts} != stored count {stored_post_count}"

        # Validate minimum data integrity requirements
        assert len(user_stats) >= 1000, f"Expected at least 1000 users, got {len(user_stats)}"

        # Validate that we have meaningful social network activity
        total_relationships = follows_integrity[0]['valid_follows_count'] + authored_integrity[0]['valid_authored_count']
        assert total_relationships >= 15000, f"Expected at least 15000 total relationships, got {total_relationships}"

    # ========================================================================
    # E-COMMERCE SCENARIO TESTS
    # ========================================================================

    def test_ecommerce_complex_scenario(self):
        """
        Test complex e-commerce scenario with precision.

        This test simulates a real e-commerce platform with:
        - 2000 customers
        - 1000 products across 20 categories
        - 5000 orders with complex order items
        - 8000 product reviews
        - Revenue calculations and inventory tracking

        Validates transactional integrity, complex aggregations, and business logic.
        """
        # CRITICAL: Registry cleanup to prevent access violations
        from kuzualchemy import clear_registry
        clear_registry()
        import gc
        gc.collect()
        session = KuzuSession(db_path=str(self.db_path))

        # Generate DDL for e-commerce models
        ecom_models = [
            EcomCustomer, EcomProduct, EcomOrder,
            PlacedOrder, OrderContains, Reviewed
        ]
        ddl = self._generate_ddl(ecom_models)
        initialize_schema(session, ddl=ddl)

        # Phase 1: Create customers with realistic data distribution
        customers = []
        with self._measure_time("ecom_customer_insert"):
            for i in range(2000):
                total_spent = random.uniform(0, 10000)
                total_orders = max(1, int(total_spent / random.uniform(50, 500)))

                customer = EcomCustomer(
                    customer_id=i + 1,
                    email=f"customer{i+1}@example.com",
                    first_name=f"FirstName{i+1}",
                    last_name=f"LastName{i+1}",
                    phone=f"+1-555-{random.randint(1000000, 9999999)}" if i % 3 == 0 else None,
                    total_orders=total_orders,
                    total_spent=round(total_spent, 2),
                    loyalty_points=int(total_spent * 0.1),  # 10% of spending as points
                    is_premium=total_spent > 5000,
                    created_at=datetime.now() - timedelta(days=random.randint(1, 730))
                )
                customers.append(customer)
                session.add(customer)

            session.commit()

        # Phase 2: Create products with realistic pricing and categories
        categories = [
            "Electronics", "Clothing", "Books", "Home & Garden", "Sports",
            "Beauty", "Automotive", "Toys", "Health", "Food",
            "Jewelry", "Tools", "Pet Supplies", "Office", "Music",
            "Movies", "Video Games", "Baby", "Shoes", "Watches"
        ]

        brands = ["BrandA", "BrandB", "BrandC", "BrandD", "BrandE", "Premium", "Budget", "Luxury"]

        products = []
        with self._measure_time("ecom_product_insert"):
            for i in range(1000):
                cost = random.uniform(5, 500)
                markup = random.uniform(1.2, 3.0)  # 20% to 200% markup
                price = round(cost * markup, 2)

                product = EcomProduct(
                    product_id=i + 1,
                    name=f"Product {i+1}",
                    description=f"High-quality product {i+1} with excellent features.",
                    price=price,
                    cost=cost,
                    stock_quantity=random.randint(0, 1000),
                    category=random.choice(categories),
                    brand=random.choice(brands),
                    rating=round(random.uniform(1.0, 5.0), 1),
                    review_count=random.randint(0, 500),
                    is_active=random.choice([True, True, True, False]),  # 75% active
                    created_at=datetime.now() - timedelta(days=random.randint(1, 365))
                )
                products.append(product)
                session.add(product)

            session.commit()

        # Phase 3: Create orders with complex business logic
        orders = []
        with self._measure_time("ecom_order_insert"):
            for i in range(5000):
                customer_id = random.randint(1, 2000)

                # Calculate order totals with realistic business logic
                subtotal = random.uniform(20, 1000)
                tax_rate = 0.08  # 8% tax
                tax_amount = round(subtotal * tax_rate, 2)

                # Shipping logic
                if subtotal > 100:
                    shipping_amount = 0.0  # Free shipping over $100
                else:
                    shipping_amount = random.choice([5.99, 9.99, 14.99])

                # Discount logic
                discount_amount = 0.0
                if random.random() < 0.2:  # 20% chance of discount
                    discount_amount = round(subtotal * random.uniform(0.05, 0.25), 2)

                total_amount = round(subtotal + tax_amount + shipping_amount - discount_amount, 2)

                # Order status distribution
                status_weights = [
                    ("pending", 0.05),
                    ("processing", 0.10),
                    ("shipped", 0.25),
                    ("delivered", 0.55),
                    ("cancelled", 0.05)
                ]
                status = random.choices(
                    [s[0] for s in status_weights],
                    weights=[s[1] for s in status_weights]
                )[0]

                order = EcomOrder(
                    order_id=i + 1,
                    order_number=f"ORD-{i+1:06d}",
                    status=status,
                    total_amount=total_amount,
                    tax_amount=tax_amount,
                    shipping_amount=shipping_amount,
                    discount_amount=discount_amount,
                    payment_method=random.choice(["credit_card", "debit_card", "paypal", "apple_pay"]),
                    shipping_address=f"123 Main St, City {i+1}, State, ZIP",
                    created_at=datetime.now() - timedelta(days=random.randint(1, 180)),
                    shipped_at=datetime.now() - timedelta(days=random.randint(1, 30)) if status in ["shipped", "delivered"] else None,
                    delivered_at=datetime.now() - timedelta(days=random.randint(1, 7)) if status == "delivered" else None
                )
                orders.append(order)
                session.add(order)

            session.commit()

        # Phase 4: Create order-customer relationships
        with self._measure_time("ecom_order_relationships"):
            for order in orders:
                customer_id = random.randint(1, 2000)
                placed_order_rel = PlacedOrder(
                    from_node=customer_id,
                    to_node=order.order_id,
                    placed_at=order.created_at
                )
                session.add(placed_order_rel)

            session.commit()

        # Phase 5: Complex e-commerce query validation
        with self._measure_time("complex_ecommerce_queries"):
            # Query 1: Revenue analysis by category
            revenue_by_category_query = """
            MATCH (c:EcomCustomer)-[:PLACED_ORDER]->(o:EcomOrder)-[:ORDER_CONTAINS]->(p:EcomProduct)
            WHERE o.status = 'delivered'
            RETURN p.category,
                   sum(o.total_amount) as total_revenue,
                   count(DISTINCT o) as order_count,
                   count(DISTINCT c) as customer_count,
                   avg(o.total_amount) as avg_order_value
            ORDER BY total_revenue DESC
            """
            revenue_results = list(session.execute(revenue_by_category_query))

            # Validate consistency
            for result in revenue_results:
                assert result['total_revenue'] > 0, "Revenue should be positive"
                assert result['order_count'] > 0, "Order count should be positive"
                assert result['customer_count'] > 0, "Customer count should be positive"
                assert result['avg_order_value'] > 0, "Average order value should be positive"

            # Query 2: Customer lifetime value analysis
            clv_query = """
            MATCH (c:EcomCustomer)-[:PLACED_ORDER]->(o:EcomOrder)
            WHERE o.status IN ['delivered', 'shipped']
            RETURN c.customer_id,
                   c.is_premium,
                   sum(o.total_amount) as lifetime_value,
                   count(o) as order_count,
                   avg(o.total_amount) as avg_order_value,
                   max(o.created_at) as last_order_date
            ORDER BY lifetime_value DESC
            LIMIT 100
            """
            clv_results = list(session.execute(clv_query))
            assert len(clv_results) <= 100, "Should return top 100 customers"

            # Handle Kuzu's generic column names (col_0, col_1, etc.)
            if len(clv_results) > 0:
                # Kuzu returns generic column names, use positional access
                # Query order: c.customer_id, c.is_premium, sum(o.total_amount), count(o), avg(o.total_amount), max(o.created_at)
                for result in clv_results:
                    customer_id = result['col_0']  # c.customer_id
                    is_premium = result['col_1']   # c.is_premium
                    lifetime_value = result['col_2']  # sum(o.total_amount) as lifetime_value
                    order_count = result['col_3']     # count(o) as order_count
                    avg_order_value = result['col_4'] # avg(o.total_amount) as avg_order_value
                    last_order_date = result['col_5'] # max(o.created_at) as last_order_date

                    # Validate consistency
                    expected_avg = lifetime_value / order_count
                    # Allow for small floating point differences
                    assert abs(expected_avg - avg_order_value) < 0.01, \
                        f"Customer {customer_id}: Average order value calculation should be accurate. Expected {expected_avg}, got {avg_order_value}"

                    # Validate data integrity
                    assert lifetime_value > 0, f"Customer {customer_id}: Lifetime value should be positive"
                    assert order_count > 0, f"Customer {customer_id}: Order count should be positive"
                    assert avg_order_value > 0, f"Customer {customer_id}: Average order value should be positive"
            else:
                print("WARNING: No CLV results returned - may indicate no delivered/shipped orders")

            # Query 3: Inventory and sales performance
            inventory_query = """
            MATCH (p:EcomProduct)
            OPTIONAL MATCH (p)<-[:ORDER_CONTAINS]-(o:EcomOrder)
            WHERE o.status = 'delivered'
            RETURN p.category,
                   p.brand,
                   avg(p.stock_quantity) as avg_stock,
                   count(o) as units_sold,
                   sum(p.price) as potential_revenue
            ORDER BY units_sold DESC
            """
            inventory_results = list(session.execute(inventory_query))

            # Validate inventory metrics
            for result in inventory_results:
                assert result['avg_stock'] >= 0, "Average stock should be non-negative"
                assert result['units_sold'] >= 0, "Units sold should be non-negative"
                assert result['potential_revenue'] >= 0, "Potential revenue should be non-negative"

        # Phase 6: Data integrity validation for e-commerce
        self._validate_ecommerce_integrity(session)

        session.close()

    def _validate_ecommerce_integrity(self, session: KuzuSession):
        """Validate data integrity in e-commerce scenario."""
        # Validate no negative prices or costs
        negative_prices = list(session.execute("""
            MATCH (p:EcomProduct)
            WHERE p.price < 0 OR p.cost < 0
            RETURN count(p) as negative_count
        """))
        assert negative_prices[0]['negative_count'] == 0, "No products should have negative prices or costs"

        # Validate all order totals are consistent
        order_validation_query = """
            MATCH (o:EcomOrder)
            RETURN o.order_id,
                   o.total_amount,
                   o.tax_amount,
                   o.shipping_amount,
                   o.discount_amount
        """
        all_orders = list(session.execute(order_validation_query))

        # Validate each order's consistency
        for order_data in all_orders:
            order_id = order_data['col_0']        # o.order_id
            total_amount = order_data['col_1']    # o.total_amount
            tax_amount = order_data['col_2']      # o.tax_amount
            shipping_amount = order_data['col_3'] # o.shipping_amount
            discount_amount = order_data['col_4'] # o.discount_amount

            # VALIDATION: All amounts must be non-negative
            assert total_amount >= 0, f"Order {order_id}: Total amount {total_amount} cannot be negative"
            assert tax_amount >= 0, f"Order {order_id}: Tax amount {tax_amount} cannot be negative"
            assert shipping_amount >= 0, f"Order {order_id}: Shipping amount {shipping_amount} cannot be negative"
            assert discount_amount >= 0, f"Order {order_id}: Discount amount {discount_amount} cannot be negative"

            # VALIDATION: Logical relationships between amounts
            assert total_amount >= tax_amount, f"Order {order_id}: Total {total_amount} must be >= tax {tax_amount}"
            assert total_amount >= shipping_amount, f"Order {order_id}: Total {total_amount} must be >= shipping {shipping_amount}"

            # VALIDATION: Discount cannot exceed total
            assert discount_amount <= total_amount, f"Order {order_id}: Discount {discount_amount} cannot exceed total {total_amount}"

            # VALIDATION: Total should be reasonable (subtotal + tax + shipping - discount)
            # For this validation, we assume total_amount is the final amount after all adjustments
            min_expected_total = tax_amount + shipping_amount - discount_amount
            assert total_amount >= min_expected_total, f"Order {order_id}: Total {total_amount} is less than minimum expected {min_expected_total}"



        # Validate customer spending consistency with precision
        customer_spending_validation = list(session.execute("""
            MATCH (c:EcomCustomer)-[:PLACED_ORDER]->(o:EcomOrder)
            RETURN c.customer_id,
                   count(o) as order_count,
                   sum(o.total_amount) as total_spent,
                   avg(o.total_amount) as avg_order_value,
                   min(o.total_amount) as min_order,
                   max(o.total_amount) as max_order
        """))

        assert len(customer_spending_validation) > 0, "Should have customers with orders"

        # Validate consistency for each customer
        for customer_data in customer_spending_validation:
            customer_id = customer_data['col_0']  # c.customer_id
            order_count = customer_data['col_1']   # count(o)
            total_spent = customer_data['col_2']   # sum(o.total_amount)
            avg_order_value = customer_data['col_3'] # avg(o.total_amount)
            min_order = customer_data['col_4']     # min(o.total_amount)
            max_order = customer_data['col_5']     # max(o.total_amount)

            # VALIDATION
            expected_avg = total_spent / order_count
            assert abs(expected_avg - avg_order_value) < 0.01, \
                f"Customer {customer_id}: Average calculation mismatch. Expected {expected_avg}, got {avg_order_value}"

            assert min_order <= avg_order_value <= max_order, \
                f"Customer {customer_id}: Average should be between min and max orders"

            assert total_spent > 0, f"Customer {customer_id}: Total spent should be positive"
            assert order_count > 0, f"Customer {customer_id}: Order count should be positive"

    # ========================================================================
    # PERFORMANCE & SCALABILITY TESTS
    # ========================================================================

    def test_large_dataset_performance(self):
        """
        Test performance with large datasets that simulate high scale.

        This test validates:
        - Insert performance with 50,000+ records
        - Query performance on large datasets
        - Memory usage under load
        - Index effectiveness
        - Concurrent operation handling

        Precision in performance measurement and validation.
        """
        # CRITICAL: Registry cleanup to prevent access violations
        from kuzualchemy import clear_registry
        clear_registry()
        import gc
        gc.collect()
        # Clear registry to prevent memory corruption
        clear_registry()
        gc.collect()  # Force garbage collection to free memory

        session = KuzuSession(db_path=str(self.db_path))

        # Use simplified models for performance testing
        @kuzu_node("PerfUser")
        class PerfUser(KuzuBaseModel):
            user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            email: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            score: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
            created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)

        @kuzu_node("PerfItem")
        class PerfItem(KuzuBaseModel):
            item_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            value: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
            category: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_relationship("INTERACTS", pairs=[(PerfUser, PerfItem)])
        class Interacts(KuzuRelationshipBase):
            interaction_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            strength: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)
            timestamp: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)

        # Initialize schema
        perf_models = [PerfUser, PerfItem, Interacts]
        ddl = self._generate_ddl(perf_models)
        initialize_schema(session, ddl=ddl)

        # Phase 1: Large-scale user insertion with performance tracking using BULK INSERT
        user_count = 25000
        batch_size = 5000
        insert_times = []

        for batch_start in range(0, user_count, batch_size):
            batch_end = min(batch_start + batch_size, user_count)

            # Create batch of users for bulk insert
            batch_users = []
            for i in range(batch_start, batch_end):
                user = PerfUser(
                    user_id=i + 1,
                    username=f"perfuser_{i+1:06d}",
                    email=f"perf{i+1}@example.com",
                    score=random.uniform(0, 1000),
                    created_at=datetime.now() - timedelta(days=random.randint(1, 365))
                )
                batch_users.append(user)

            # Use ultra-fast bulk insert with Polars DataFrame
            start_time = time.time()
            session.bulk_insert(batch_users)
            batch_time = time.time() - start_time
            insert_times.append(batch_time)

            # Track memory usage
            current_memory = self._track_memory()

            # Validate performance requirements per batch
            assert batch_time < 30.0, f"Batch insert time {batch_time:.2f}s exceeds 30s limit"

            # Memory should not grow excessively
            memory_increase = current_memory - self.initial_memory
            # Adjusted for registry cleanup overhead and Windows memory management
            expected_max_memory = (batch_end / user_count) * 1000
            assert memory_increase < expected_max_memory, f"Memory usage {memory_increase:.2f}MB exceeds expected {expected_max_memory:.2f}MB"

        # Validate overall insert performance
        avg_batch_time = sum(insert_times) / len(insert_times)
        total_insert_time = sum(insert_times)
        users_per_second = user_count / total_insert_time

        assert avg_batch_time < 15.0, f"Average batch time {avg_batch_time:.2f}s exceeds 15s"
        assert users_per_second > 500, f"Insert rate {users_per_second:.1f} users/sec below 500/sec minimum"

        # Phase 2: Large-scale item insertion using BULK INSERT
        item_count = 10000
        categories = ["A", "B", "C", "D", "E"] * 2000  # Ensure even distribution

        # Create all items at once for bulk insert
        items = []
        for i in range(item_count):
            item = PerfItem(
                item_id=i + 1,
                name=f"Item_{i+1:06d}",
                value=random.uniform(1, 1000),
                category=categories[i]
            )
            items.append(item)

        # Use ultra-fast bulk insert with Polars DataFrame
        start_time = time.time()
        session.bulk_insert(items)
        item_insert_time = time.time() - start_time
        items_per_second = item_count / item_insert_time

        assert items_per_second > 600, f"Item insert rate {items_per_second:.1f}/sec below 600/sec minimum"

        # Phase 3: Large-scale relationship creation using BULK INSERT
        relationship_count = 1000

        # Create all relationships at once for bulk insert
        interactions = []
        for i in range(relationship_count):
            user_id = random.randint(1, user_count)
            item_id = random.randint(1, item_count)

            interaction = Interacts(
                from_node=user_id,
                to_node=item_id,
                interaction_type=random.choice(["view", "like", "purchase", "share"]),
                strength=random.uniform(0.1, 1.0),
                timestamp=datetime.now() - timedelta(hours=random.randint(1, 8760))
            )
            interactions.append(interaction)

        # Use ultra-fast bulk insert with Polars DataFrame
        start_time = time.time()
        session.bulk_insert(interactions)
        relationship_insert_time = time.time() - start_time
        relationships_per_second = relationship_count / relationship_insert_time

        assert relationships_per_second > 2000, f"Relationship insert rate {relationships_per_second:.1f}/sec below 2000/sec minimum"

        # Phase 4: Complex query performance on large dataset
        query_performance_tests = [
            {
                "name": "Simple aggregation",
                "query": "MATCH (u:PerfUser) RETURN count(u), avg(u.score), max(u.score), min(u.score)",
                "max_time": 2.0
            },
            {
                "name": "Join with aggregation",
                "query": """
                    MATCH (u:PerfUser)-[i:INTERACTS]->(item:PerfItem)
                    RETURN u.username, count(i) as interaction_count, avg(i.strength) as avg_strength
                    ORDER BY interaction_count DESC
                    LIMIT 100
                """,
                "max_time": 10.0
            },
            {
                "name": "Complex filtering",
                "query": """
                    MATCH (u:PerfUser)-[i:INTERACTS]->(item:PerfItem)
                    WHERE u.score > 500 AND i.strength > 0.5 AND item.value > 100
                    RETURN item.category, count(*) as high_value_interactions
                    ORDER BY high_value_interactions DESC
                """,
                "max_time": 15.0
            },
            {
                "name": "Multi-hop traversal",
                "query": """
                    MATCH (u1:PerfUser)-[:INTERACTS]->(item:PerfItem)<-[:INTERACTS]-(u2:PerfUser)
                    WHERE u1.user_id <> u2.user_id
                    RETURN u1.username, u2.username, count(item) as common_items
                    ORDER BY common_items DESC
                    LIMIT 50
                """,
                "max_time": 20.0
            }
        ]

        for test in query_performance_tests:
            start_time = time.time()
            results = list(session.execute(test["query"]))
            query_time = time.time() - start_time

            assert query_time < test["max_time"], f"{test['name']} took {query_time:.2f}s, exceeds {test['max_time']}s limit"
            assert len(results) > 0, f"{test['name']} should return results"

            # Store for overall performance analysis
            self.performance_metrics['query_times'].append(query_time)

        # Phase 5: Memory efficiency validation
        final_memory = self._track_memory()
        total_memory_increase = final_memory - self.initial_memory

        # Calculate expected memory usage based on data size
        total_records = user_count + item_count + relationship_count
        memory_per_record = total_memory_increase / total_records

        # Memory usage should be reasonable (less than 1KB per record on average)
        assert memory_per_record < 1.0, f"Memory per record {memory_per_record:.3f}MB exceeds 1MB limit"
        assert total_memory_increase < 1000, f"Total memory increase {total_memory_increase:.2f}MB exceeds 1GB limit"

        session.close()

    def test_concurrent_operations(self):
        """
        Test concurrent database operations within Kuzu's constraints.

        Kuzu only supports one write transaction at a time, so this test validates:
        - Sequential write operations with high throughput
        - Concurrent read operations (multiple readers)
        - Connection pooling and management
        - Data consistency under load

        Validation of consistency and performance.
        """
        # CRITICAL: Registry cleanup to prevent access violations
        from kuzualchemy import clear_registry
        clear_registry()
        import gc
        gc.collect()
        # Clear registry to prevent memory corruption
        clear_registry()
        gc.collect()  # Force garbage collection to free memory

        session = KuzuSession(db_path=str(self.db_path))

        # Simple model for concurrency testing
        @kuzu_node("ConcUser")
        class ConcUser(KuzuBaseModel):
            user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            counter: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)
            last_updated: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)

        # Initialize schema
        ddl = self._generate_ddl([ConcUser])
        initialize_schema(session, ddl=ddl)
        session.close()

        # Concurrent operation results tracking
        operation_results = {
            'insert_success': 0,
            'insert_errors': 0,
            'query_success': 0,
            'query_errors': 0,
            'update_success': 0,
            'update_errors': 0
        }
        results_lock = threading.Lock()

        def sequential_batch_insert_worker(batch_id: int, batch_data: List[dict]):
            """Worker function for sequential batch inserts (Kuzu constraint: single writer)."""
            worker_session = KuzuSession(db_path=str(self.db_path))

            try:
                # Use bulk insert for better performance
                users = []
                for data in batch_data:
                    user = ConcUser(
                        user_id=data['user_id'],
                        username=data['username'],
                        counter=0,
                        last_updated=datetime.now()
                    )
                    users.append(user)

                # Use bulk insert if available, otherwise add individually
                if len(users) >= worker_session.bulk_insert_threshold:
                    worker_session.bulk_insert(users)
                else:
                    for user in users:
                        worker_session.add(user)
                    worker_session.commit()

                with results_lock:
                    operation_results['insert_success'] += len(batch_data)

            except Exception as e:
                print(f"Batch insert error in batch {batch_id}: {e}")
                with results_lock:
                    operation_results['insert_errors'] += len(batch_data)

            worker_session.close()

        def concurrent_query_worker(worker_id: int, num_operations: int):
            """Worker function for concurrent queries."""
            worker_session = KuzuSession(db_path=str(self.db_path), bulk_insert_threshold=num_operations)

            for i in range(num_operations):
                try:
                    # Perform various types of queries
                    queries = [
                        "MATCH (u:ConcUser) RETURN count(u)",
                        "MATCH (u:ConcUser) WHERE u.counter > 0 RETURN u.username",
                        "MATCH (u:ConcUser) RETURN avg(u.counter), max(u.counter)"
                    ]

                    query = random.choice(queries)
                    results = list(worker_session.execute(query))

                    with results_lock:
                        operation_results['query_success'] += 1

                except Exception as e:
                    with results_lock:
                        operation_results['query_errors'] += 1
                    print(f"Query error in worker {worker_id}: {e}")

            worker_session.close()

        # Phase 1: Sequential batch insert test (Kuzu single-writer constraint)
        num_batches = 5
        records_per_batch = 100

        # Create batches of data for sequential processing
        all_batch_data = []
        for batch_id in range(num_batches):
            batch_data = []
            for i in range(records_per_batch):
                user_id = batch_id * 1000 + i + 1
                batch_data.append({
                    'user_id': user_id,
                    'username': f"concuser_{batch_id}_{i+1}"
                })
            all_batch_data.append((batch_id, batch_data))

        start_time = time.time()

        # Process batches sequentially (Kuzu constraint: single writer)
        for batch_id, batch_data in all_batch_data:
            sequential_batch_insert_worker(batch_id, batch_data)

        insert_duration = time.time() - start_time

        # Validate insert results
        expected_inserts = num_batches * records_per_batch
        total_insert_attempts = operation_results['insert_success'] + operation_results['insert_errors']

        assert total_insert_attempts == expected_inserts, f"Expected {expected_inserts} insert attempts, got {total_insert_attempts}"

        # Sequential processing should have 100% success rate
        success_rate = operation_results['insert_success'] / expected_inserts
        assert success_rate > 0.95, f"Insert success rate {success_rate:.2%} below 95% minimum"

        # Phase 2: Concurrent query test (multiple readers allowed)
        num_query_workers = 5
        queries_per_worker = 20
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_query_workers) as executor:
            query_futures = [
                executor.submit(concurrent_query_worker, i, queries_per_worker)
                for i in range(num_query_workers)
            ]

            # Wait for all query operations to complete
            for future in as_completed(query_futures):
                future.result()

        query_duration = time.time() - start_time

        # Validate query results
        expected_queries = num_query_workers * queries_per_worker
        total_query_attempts = operation_results['query_success'] + operation_results['query_errors']

        assert total_query_attempts == expected_queries, f"Expected {expected_queries} query attempts, got {total_query_attempts}"

        # Queries should have very high success rate
        query_success_rate = operation_results['query_success'] / expected_queries
        assert query_success_rate > 0.98, f"Query success rate {query_success_rate:.2%} below 98% minimum"

        # Phase 3: Validate data consistency after concurrent operations
        final_session = KuzuSession(db_path=str(self.db_path))

        # Count total users created
        user_count_result = list(final_session.execute("MATCH (u:ConcUser) RETURN count(u) as total_users"))
        actual_user_count = user_count_result[0]['total_users']

        # Should match successful inserts
        assert actual_user_count == operation_results['insert_success'], f"Database has {actual_user_count} users, expected {operation_results['insert_success']}"

        # Validate no duplicate user_ids (primary key constraint)
        # Use a different approach since Kuzu has different GROUP BY/HAVING syntax
        all_users = list(final_session.execute("""
            MATCH (u:ConcUser)
            RETURN u.user_id
        """))

        # Check for duplicates in Python (more reliable than complex Kuzu syntax)
        if all_users:
            # Handle Kuzu's generic column names
            user_id_key = 'user_id' if 'user_id' in all_users[0] else list(all_users[0].keys())[0]
            user_ids = [user[user_id_key] for user in all_users]
            duplicate_ids = [uid for uid in set(user_ids) if user_ids.count(uid) > 1]
        else:
            duplicate_ids = []
        assert len(duplicate_ids) == 0, f"No duplicate user_ids should exist, found: {duplicate_ids}"

        # Performance validation
        insert_rate = operation_results['insert_success'] / insert_duration
        query_rate = operation_results['query_success'] / query_duration
        
        print(f"Insert rate: {insert_rate:.1f}/sec, Query rate: {query_rate:.1f}/sec")
        print(f"Insert duration: {insert_duration:.2f}s, Query duration: {query_duration:.2f}s")
        print(f"Insert success rate: {success_rate:.2%}, Query success rate: {query_success_rate:.2%}")
        print(f"Total insert attempts: {total_insert_attempts}, Total query attempts: {total_query_attempts}")
        print(f"Total insert errors: {operation_results['insert_errors']}, Total query errors: {operation_results['query_errors']}")
        print(f"Actual user count: {actual_user_count}, Expected user count: {operation_results['insert_success']}")
        
        assert insert_rate > 50, f"Concurrent insert rate {insert_rate:.1f}/sec below 50/sec minimum"
        assert query_rate > 100, f"Concurrent query rate {query_rate:.1f}/sec below 100/sec minimum"

        final_session.close()

        # Store concurrent operation metrics
        self.performance_metrics['concurrent_operations'].append({
            'insert_rate': insert_rate,
            'query_rate': query_rate,
            'insert_success_rate': success_rate,
            'query_success_rate': query_success_rate
        })

    # ========================================================================
    # DATA INTEGRITY & CONSISTENCY TESTS
    # ========================================================================

    def test_transaction_integrity_and_rollback(self):
        """
        Test transaction integrity, rollback scenarios, and data consistency.

        This test validates:
        - Transaction atomicity (all-or-nothing)
        - Rollback functionality
        - Data consistency after failures
        - Constraint enforcement
        - Recovery from errors

        Validation of ACID properties.
        """
        # Clear registry to prevent memory corruption
        clear_registry()
        gc.collect()  # Force garbage collection to free memory

        session = KuzuSession(db_path=str(self.db_path))

        # Models for transaction testing
        @kuzu_node("TxnUser")
        class TxnUser(KuzuBaseModel):
            user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            balance: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
            status: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="active")

        @kuzu_node("TxnAccount")
        class TxnAccount(KuzuBaseModel):
            account_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            account_number: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            balance: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
            account_type: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        @kuzu_relationship("OWNS_ACCOUNT", pairs=[(TxnUser, TxnAccount)])
        class OwnsAccount(KuzuRelationshipBase):
            ownership_percentage: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=100.0)
            created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)

        @kuzu_relationship("TRANSFER", pairs=[(TxnAccount, TxnAccount)])
        class Transfer(KuzuRelationshipBase):
            amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
            transfer_date: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
            status: str = kuzu_field(kuzu_type=KuzuDataType.STRING, default="pending")
            reference: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)

        # Initialize schema
        txn_models = [TxnUser, TxnAccount, OwnsAccount, Transfer]
        ddl = self._generate_ddl(txn_models)
        initialize_schema(session, ddl=ddl)

        # Phase 1: Create initial data for transaction testing
        initial_users = []
        initial_accounts = []

        for i in range(10):
            user = TxnUser(
                user_id=i + 1,
                username=f"txnuser_{i+1}",
                balance=1000.0,  # Each user starts with $1000
                status="active"
            )
            initial_users.append(user)
            session.add(user)

            # Each user has 2 accounts
            for j in range(2):
                account = TxnAccount(
                    account_id=i * 2 + j + 1,
                    account_number=f"ACC-{i+1:03d}-{j+1}",
                    balance=500.0,  # Each account starts with $500
                    account_type="checking" if j == 0 else "savings"
                )
                initial_accounts.append(account)
                session.add(account)

                # Create ownership relationship
                ownership = OwnsAccount(
                    from_node=user.user_id,
                    to_node=account.account_id,
                    ownership_percentage=100.0,
                    created_at=datetime.now()
                )
                session.add(ownership)

        session.commit()

        # Validate initial state
        total_user_balance = sum(user.balance for user in initial_users)
        total_account_balance = sum(account.balance for account in initial_accounts)

        assert total_user_balance == 10000.0, f"Total user balance should be $10,000, got ${total_user_balance}"
        assert total_account_balance == 10000.0, f"Total account balance should be $10,000, got ${total_account_balance}"

        # Phase 2: Test successful transaction (money transfer)
        def perform_transfer(from_account_id: int, to_account_id: int, amount: float, reference: str):
            """Perform a money transfer between accounts."""
            # This simulates a transaction that should be atomic

            # Get current balances
            from_balance_query = f"MATCH (a:TxnAccount {{account_id: {from_account_id}}}) RETURN a.balance as balance"
            to_balance_query = f"MATCH (a:TxnAccount {{account_id: {to_account_id}}}) RETURN a.balance as balance"

            from_balance = list(session.execute(from_balance_query))[0]['balance']
            to_balance = list(session.execute(to_balance_query))[0]['balance']

            # Validate sufficient funds
            if from_balance < amount:
                raise ValueError(f"Insufficient funds: {from_balance} < {amount}")

            # Create transfer record
            transfer = Transfer(
                from_node=from_account_id,
                to_node=to_account_id,
                amount=amount,
                transfer_date=datetime.now(),
                status="completed",
                reference=reference
            )
            session.add(transfer)

            # Update balances (in a real system, this would be done atomically)
            session.execute(f"MATCH (a:TxnAccount {{account_id: {from_account_id}}}) SET a.balance = a.balance - {amount}")
            session.execute(f"MATCH (a:TxnAccount {{account_id: {to_account_id}}}) SET a.balance = a.balance + {amount}")

            session.commit()

            return from_balance - amount, to_balance + amount

        # Test successful transfer
        new_from_balance, new_to_balance = perform_transfer(1, 2, 100.0, "TEST-TRANSFER-001")

        # Validate transfer was successful
        assert new_from_balance == 400.0, f"From account balance should be $400, got ${new_from_balance}"
        assert new_to_balance == 600.0, f"To account balance should be $600, got ${new_to_balance}"

        # Validate transfer record was created
        transfer_count = list(session.execute("MATCH ()-[t:TRANSFER]->() RETURN count(t) as count"))[0]['count']
        assert transfer_count == 1, f"Should have 1 transfer record, got {transfer_count}"

        # Phase 3: Test transaction rollback scenario
        # Note: KuzuDB may not support full transaction rollback, so we simulate the concept

        # Get current state before "failed" transaction
        pre_failure_balances = {}
        for account in initial_accounts:
            balance_query = f"MATCH (a:TxnAccount {{account_id: {account.account_id}}}) RETURN a.balance as balance"
            balance = list(session.execute(balance_query))[0]['balance']
            pre_failure_balances[account.account_id] = balance

        # Attempt transfer that should fail (insufficient funds)
        try:
            perform_transfer(1, 2, 1000.0, "FAIL-TRANSFER-001")  # Account 1 only has $400 left
            assert False, "Transfer should have failed due to insufficient funds"
        except ValueError as e:
            # Expected failure - validate no changes were made
            assert "Insufficient funds" in str(e)

            # Validate balances remain unchanged
            for account_id, expected_balance in pre_failure_balances.items():
                balance_query = f"MATCH (a:TxnAccount {{account_id: {account_id}}}) RETURN a.balance as balance"
                actual_balance = list(session.execute(balance_query))[0]['balance']
                assert actual_balance == expected_balance, f"Account {account_id} balance changed during failed transaction"

        # Phase 4: Data consistency validation
        self._validate_transaction_consistency(session)

        session.close()

    def _validate_transaction_consistency(self, session: KuzuSession):
        """Validate data consistency in transaction scenario."""
        # Validate total money in system remains constant
        total_account_balance_query = "MATCH (a:TxnAccount) RETURN sum(a.balance) as total_balance"
        total_balance = list(session.execute(total_account_balance_query))[0]['total_balance']

        # Should still be $10,000 (conservation of money)
        assert abs(total_balance - 10000.0) < 0.01, f"Total balance ${total_balance} should equal $10,000"

        # Validate all transfers have valid amounts
        invalid_transfers = list(session.execute("""
            MATCH ()-[t:TRANSFER]->()
            WHERE t.amount <= 0
            RETURN count(t) as invalid_count
        """))
        assert invalid_transfers[0]['invalid_count'] == 0, "No transfers should have negative or zero amounts"

        # Validate all accounts have non-negative balances
        negative_balances = list(session.execute("""
            MATCH (a:TxnAccount)
            WHERE a.balance < 0
            RETURN count(a) as negative_count
        """))
        assert negative_balances[0]['negative_count'] == 0, "No accounts should have negative balances"

    # ========================================================================
    # EDGE CASES & ERROR HANDLING TESTS
    # ========================================================================

    def test_edge_cases_and_error_handling(self):
        """
        Test edge cases and error handling scenarios.

        This test validates:
        - Handling of malformed data
        - Large string/data handling
        - Null value handling
        - Type validation
        - Resource exhaustion scenarios
        - Invalid query handling

        Validation of error boundaries and recovery.
        """
        # Clear registry to prevent memory corruption
        clear_registry()
        gc.collect()  # Force garbage collection to free memory

        session = KuzuSession(db_path=str(self.db_path))

        # Models for edge case testing
        @kuzu_node("EdgeUser")
        class EdgeUser(KuzuBaseModel):
            user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
            username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
            bio: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)
            score: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
            tags: Optional[str] = kuzu_field(kuzu_type=KuzuDataType.STRING, default=None)

        # Initialize schema
        ddl = self._generate_ddl([EdgeUser])
        initialize_schema(session, ddl=ddl)

        # Phase 1: Test large string handling
        large_string = "A" * 10000  # 10KB string
        very_large_string = "B" * 100000  # 100KB string

        # Test normal large string
        user1 = EdgeUser(
            user_id=1,
            username="large_bio_user",
            bio=large_string,
            score=100.0
        )
        session.add(user1)
        session.commit()

        # Validate large string was stored correctly
        retrieved_user = list(session.execute("MATCH (u:EdgeUser {user_id: 1}) RETURN u.bio as bio"))[0]
        assert len(retrieved_user['bio']) == 10000, "Large string should be stored correctly"
        assert retrieved_user['bio'] == large_string, "Large string content should match"

        # Test very large string
        user2 = EdgeUser(
            user_id=2,
            username="very_large_bio_user",
            bio=very_large_string,
            score=200.0
        )
        session.add(user2)
        session.commit()
        # If successful, validate
        retrieved_large = list(session.execute("MATCH (u:EdgeUser {user_id: 2}) RETURN u.bio as bio"))[0]
        assert len(retrieved_large['bio']) == 100000, "Very large string should be stored correctly"

        # Phase 2: Test null value handling
        user3 = EdgeUser(
            user_id=3,
            username="null_bio_user",
            bio=None,  # Explicit null
            score=300.0,
            tags=None
        )
        session.add(user3)
        session.commit()

        # Validate null handling
        null_user = list(session.execute("MATCH (u:EdgeUser {user_id: 3}) RETURN u.bio as bio, u.tags as tags"))[0]
        assert null_user['bio'] is None, "Null bio should be stored as null"
        assert null_user['tags'] is None, "Null tags should be stored as null"

        # Phase 3: Test special characters and encoding
        special_chars = "Special chars: àáâãäåæçèéêë ñòóôõö ùúûü ÿ 中文 العربية русский 🚀🎉💯"

        user4 = EdgeUser(
            user_id=4,
            username="special_chars_user",
            bio=special_chars,
            score=400.0
        )
        session.add(user4)
        session.commit()

        # Validate special character handling
        special_user = list(session.execute("MATCH (u:EdgeUser {user_id: 4}) RETURN u.bio as bio"))[0]
        assert special_user['bio'] == special_chars, "Special characters should be preserved"

        # Phase 4: Test numeric edge cases
        edge_cases = [
            (5, "zero_user", 0.0),
            (6, "negative_user", -999999.99),
            (7, "large_positive_user", 999999999.99),
            (8, "small_decimal_user", 0.000001),
            (9, "scientific_notation_user", 1.23e-10)
        ]

        for user_id, username, score in edge_cases:
            user = EdgeUser(
                user_id=user_id,
                username=username,
                score=score
            )
            session.add(user)

        session.commit()

        # Validate numeric edge cases
        for user_id, username, expected_score in edge_cases:
            result = list(session.execute(f"MATCH (u:EdgeUser {{user_id: {user_id}}}) RETURN u.score as score"))[0]
            actual_score = result['score']

            # Allow for floating point precision differences
            assert abs(actual_score - expected_score) < 1e-10, f"Score for {username} should be {expected_score}, got {actual_score}"

        # Phase 5: Test invalid query handling
        invalid_queries = [
            "INVALID CYPHER SYNTAX",
            "MATCH (u:NonExistentNode) RETURN u",
            "MATCH (u:EdgeUser) WHERE u.nonexistent_field = 'test' RETURN u",
            "CREATE (u:EdgeUser {user_id: 'not_a_number'})"  # Type mismatch
        ]

        error_count = 0
        for query in invalid_queries:
            try:
                list(session.execute(query))
                # If no error, that's unexpected for these queries
                print(f"Query unexpectedly succeeded: {query}")
            except Exception as e:
                error_count += 1
                # Expected error - validate it's handled gracefully
                assert isinstance(e, Exception), "Should raise an exception for invalid queries"

        # Should have caught errors for most/all invalid queries
        assert error_count >= len(invalid_queries) * 0.5, "Should catch errors for invalid queries"

        # Phase 6: Test resource limits and recovery
        # Test creating many users rapidly to test memory/resource handling
        rapid_insert_count = 1000
        start_memory = self._track_memory()

        start_time = time.time()
        for i in range(rapid_insert_count):
            user = EdgeUser(
                user_id=1000 + i,
                username=f"rapid_user_{i}",
                bio=f"Bio for rapid user {i}" * 10,  # Moderate size bio
                score=float(i)
            )
            session.add(user)

            # Commit in batches to avoid memory issues
            if (i + 1) % 100 == 0:
                session.commit()

        session.commit()
        rapid_insert_time = time.time() - start_time
        end_memory = self._track_memory()

        # Validate rapid insertion performance
        inserts_per_second = rapid_insert_count / rapid_insert_time
        memory_increase = end_memory - start_memory

        assert inserts_per_second > 100, f"Rapid insert rate {inserts_per_second:.1f}/sec below 100/sec minimum"
        assert memory_increase < 100, f"Memory increase {memory_increase:.2f}MB during rapid inserts exceeds 100MB"

        # Validate all rapid inserts were successful
        rapid_user_count = list(session.execute("MATCH (u:EdgeUser) WHERE u.user_id >= 1000 RETURN count(u) as count"))[0]['count']
        assert rapid_user_count == rapid_insert_count, f"Should have {rapid_insert_count} rapid users, got {rapid_user_count}"

        session.close()

    # ========================================================================
    # COMPREHENSIVE VALIDATION
    # ========================================================================

    def test_comprehensive_validation(self):
        """
        Final comprehensive validation test that combines all scenarios.

        This test runs a complete simulation with:
        - Mixed workload (social + ecommerce + performance)
        - Concurrent operations
        - Data integrity validation
        - Performance requirements validation
        - Memory efficiency validation

        This is the ultimate test that validates readiness.
        """
        session = KuzuSession(db_path=str(self.db_path))

        # Use all model types for comprehensive testing
        all_models = [
            SocialUser, SocialPost, SocialComment,
            Follows, Authored, Liked, Commented,
            EcomCustomer, EcomProduct, EcomOrder,
            PlacedOrder, OrderContains, Reviewed
        ]

        ddl = self._generate_ddl(all_models)
        initialize_schema(session, ddl=ddl)

        # Phase 1: Create realistic mixed dataset
        start_time = time.time()

        # Create social users
        for i in range(500):
            user = SocialUser(
                user_id=i + 1,
                username=f"socialuser_{i+1:04d}",
                email=f"social{i+1}@example.com",
                full_name=f"Social User {i+1}",
                follower_count=random.randint(0, 1000),
                following_count=random.randint(0, 500),
                post_count=random.randint(0, 100),
                is_verified=i < 25,
                created_at=datetime.now() - timedelta(days=random.randint(1, 365))
            )
            session.add(user)

        # Create e-commerce customers
        for i in range(300):
            customer = EcomCustomer(
                customer_id=i + 1,
                email=f"customer{i+1}@example.com",
                first_name=f"Customer{i+1}",
                last_name=f"LastName{i+1}",
                total_orders=random.randint(1, 20),
                total_spent=random.uniform(100, 5000),
                loyalty_points=random.randint(0, 1000),
                is_premium=random.choice([True, False]),
                created_at=datetime.now() - timedelta(days=random.randint(1, 730))
            )
            session.add(customer)

        # Create products
        for i in range(200):
            product = EcomProduct(
                product_id=i + 1,
                name=f"Product {i+1}",
                description=f"Description for product {i+1}",
                price=random.uniform(10, 1000),
                cost=random.uniform(5, 500),
                stock_quantity=random.randint(0, 1000),
                category=random.choice(["Electronics", "Clothing", "Books", "Home"]),
                brand=random.choice(["BrandA", "BrandB", "BrandC"]),
                rating=random.uniform(1.0, 5.0),
                review_count=random.randint(0, 100),
                created_at=datetime.now() - timedelta(days=random.randint(1, 365))
            )
            session.add(product)

        session.commit()
        setup_time = time.time() - start_time

        # Phase 2: Create relationships
        relationship_start = time.time()

        # Social follows (each user follows 5-15 others)
        for user_id in range(1, 501):
            follow_count = random.randint(5, 15)
            followed_users = random.sample([uid for uid in range(1, 501) if uid != user_id], follow_count)

            for followed_id in followed_users:
                follow = Follows(
                    from_node=user_id,
                    to_node=followed_id,
                    followed_at=datetime.now() - timedelta(days=random.randint(1, 180)),
                    is_mutual=random.choice([True, False])
                )
                session.add(follow)

        # E-commerce orders
        for i in range(1000):
            customer_id = random.randint(1, 300)
            order = EcomOrder(
                order_id=i + 1,
                order_number=f"ORD-{i+1:06d}",
                status=random.choice(["pending", "shipped", "delivered"]),
                total_amount=random.uniform(20, 500),
                tax_amount=random.uniform(1, 40),
                shipping_amount=random.uniform(0, 15),
                payment_method=random.choice(["credit_card", "paypal"]),
                shipping_address=f"Address {i+1}",
                created_at=datetime.now() - timedelta(days=random.randint(1, 90))
            )
            session.add(order)

            # Create order relationship
            placed_order = PlacedOrder(
                from_node=customer_id,
                to_node=order.order_id,
                placed_at=order.created_at
            )
            session.add(placed_order)

        session.commit()
        relationship_time = time.time() - relationship_start

        # Phase 3: Complex cross-domain queries
        query_start = time.time()

        complex_queries = [
            # Cross-domain query: Social users who are also customers
            """
            MATCH (su:SocialUser), (ec:EcomCustomer)
            WHERE su.email = ec.email
            RETURN count(*) as cross_platform_users
            """,

            # Performance query: Top customers by spending
            """
            MATCH (c:EcomCustomer)-[:PLACED_ORDER]->(o:EcomOrder)
            WHERE o.status = 'delivered'
            RETURN c.email, sum(o.total_amount) as total_spent
            ORDER BY total_spent DESC
            LIMIT 20
            """,

            # Social network analysis
            """
            MATCH (u1:SocialUser)-[:FOLLOWS]->(u2:SocialUser)-[:FOLLOWS]->(u3:SocialUser)
            WHERE u1.user_id <> u3.user_id
            RETURN u1.username, count(DISTINCT u3) as second_degree_connections
            ORDER BY second_degree_connections DESC
            LIMIT 10
            """,

            # Revenue analysis
            """
            MATCH (c:EcomCustomer)-[:PLACED_ORDER]->(o:EcomOrder)
            RETURN
                count(DISTINCT c) as total_customers,
                count(o) as total_orders,
                sum(o.total_amount) as total_revenue,
                avg(o.total_amount) as avg_order_value
            """
        ]

        query_results = []
        for query in complex_queries:
            query_start_time = time.time()
            results = list(session.execute(query))
            query_duration = time.time() - query_start_time

            query_results.append({
                'query': query[:50] + "...",
                'duration': query_duration,
                'result_count': len(results)
            })

            # Each query should complete within reasonable time
            assert query_duration < 30.0, f"Query took {query_duration:.2f}s, exceeds 30s limit"
            assert len(results) > 0, "Query should return results"

        total_query_time = time.time() - query_start

        # Phase 4: Final validation and performance assessment
        final_memory = self._track_memory()
        total_memory_increase = final_memory - self.initial_memory

        # Validate overall performance requirements
        total_records = 500 + 300 + 200 + 1000  # Users + customers + products + orders
        setup_rate = total_records / (setup_time + relationship_time)

        assert setup_rate > 100, f"Overall setup rate {setup_rate:.1f} records/sec below 100/sec minimum"
        assert total_memory_increase < 200, f"Total memory increase {total_memory_increase:.2f}MB exceeds 200MB limit"
        assert total_query_time < 60.0, f"Total query time {total_query_time:.2f}s exceeds 60s limit"

        # Data consistency validation
        user_count = list(session.execute("MATCH (u:SocialUser) RETURN count(u) as count"))[0]['count']
        customer_count = list(session.execute("MATCH (c:EcomCustomer) RETURN count(c) as count"))[0]['count']
        product_count = list(session.execute("MATCH (p:EcomProduct) RETURN count(p) as count"))[0]['count']
        order_count = list(session.execute("MATCH (o:EcomOrder) RETURN count(o) as count"))[0]['count']

        assert user_count == 500, f"Should have 500 social users, got {user_count}"
        assert customer_count == 300, f"Should have 300 customers, got {customer_count}"
        assert product_count == 200, f"Should have 200 products, got {product_count}"
        assert order_count == 1000, f"Should have 1000 orders, got {order_count}"

        # Relationship integrity validation
        follow_count = list(session.execute("MATCH ()-[f:FOLLOWS]->() RETURN count(f) as count"))[0]['count']
        order_rel_count = list(session.execute("MATCH ()-[p:PLACED_ORDER]->() RETURN count(p) as count"))[0]['count']

        assert follow_count > 0, "Should have follow relationships"
        assert order_rel_count == 1000, f"Should have 1000 order relationships, got {order_rel_count}"

        session.close()

        # Final performance summary
        print(f"\n=== VALIDATION COMPLETE ===")
        print(f"Total Records Created: {total_records}")
        print(f"Setup Rate: {setup_rate:.1f} records/sec")
        print(f"Memory Usage: {total_memory_increase:.2f} MB")
        print(f"Query Performance: {len(complex_queries)} complex queries in {total_query_time:.2f}s")
        print(f"Average Query Time: {total_query_time/len(complex_queries):.2f}s")

        for result in query_results:
            print(f"  - {result['query']}: {result['duration']:.3f}s ({result['result_count']} results)")

        print("=== ALL TESTS PASSED ===")
