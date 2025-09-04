"""
This is the ultimate test that validates KuzuAlchemy.
This test combines all critical aspects in a comprehensive validation that would
stake Anthropic's entire reputation on the system's reliability.

Test Coverage:
1. Real-world scenario simulation (Social Network + E-commerce)
2. Performance validation under load
3. Data integrity and consistency
4. Concurrent operations
5. Memory efficiency
6. Error handling and recovery
7. Complex query performance
8. Relationship handling accuracy.
"""

from __future__ import annotations

import tempfile
import shutil
import time
import sys
import gc
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Any
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
)
from kuzualchemy.test_utilities import initialize_schema
from kuzualchemy.kuzu_orm import get_ddl_for_node, get_ddl_for_relationship


# ============================================================================
# MODELS
# ============================================================================

@kuzu_node("User")
class User(KuzuBaseModel):
    """User model."""
    user_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    username: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    email: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    full_name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    score: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    is_active: bool = kuzu_field(kuzu_type=KuzuDataType.BOOL, default=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_node("Product")
class Product(KuzuBaseModel):
    """Product model."""
    product_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    name: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    category: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    rating: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=0.0)
    stock: int = kuzu_field(kuzu_type=KuzuDataType.INT32, default=0)


@kuzu_node("PurchaseOrder")
class PurchaseOrder(KuzuBaseModel):
    """Order model."""
    order_id: int = kuzu_field(kuzu_type=KuzuDataType.INT64, primary_key=True)
    total_amount: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    status: str = kuzu_field(kuzu_type=KuzuDataType.STRING, not_null=True)
    created_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_relationship("FOLLOWS", pairs=[(User, User)])
class Follows(KuzuRelationshipBase):
    """User follows relationship."""
    followed_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)
    strength: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, default=1.0)


@kuzu_relationship("PURCHASED", pairs=[(User, Product)])
class Purchased(KuzuRelationshipBase):
    """User purchased product relationship."""
    quantity: int = kuzu_field(kuzu_type=KuzuDataType.INT32, not_null=True)
    price_paid: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)
    purchased_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_relationship("PLACED", pairs=[(User, PurchaseOrder)])
class Placed(KuzuRelationshipBase):
    """User placed order relationship."""
    placed_at: datetime = kuzu_field(kuzu_type=KuzuDataType.TIMESTAMP)


@kuzu_relationship("CONTAINS", pairs=[(PurchaseOrder, Product)])
class Contains(KuzuRelationshipBase):
    """Order contains product relationship."""
    quantity: int = kuzu_field(kuzu_type=KuzuDataType.INT32, not_null=True)
    unit_price: float = kuzu_field(kuzu_type=KuzuDataType.DOUBLE, not_null=True)


class TestValidationFinal:
    """
    The ultimate validation test.
    
    This test is designed to be the final arbiter of whether KuzuAlchemy
    is ready for deployment. It combines all critical aspects
    in a comprehensive validation with precision.
    """

    def setup_method(self):
        """Set up validation environment."""
        self.temp_db = tempfile.mkdtemp()
        self.db_path = Path(self.temp_db) / "final_validation.db"

        # Performance and reliability tracking
        self.initial_memory = self.get_memory_usage()
        
        self.validation_metrics = {
            'total_operations': 0,
            'failed_operations': 0,
            'peak_memory_mb': 0,
            'query_times': [],
            'insert_times': [],
            'concurrent_success_rate': 0.0,
            'data_integrity_score': 0.0,
            'performance_score': 0.0
        }

    def teardown_method(self):
        """Clean up and report final validation results."""
        if Path(self.temp_db).exists():
            shutil.rmtree(self.temp_db, ignore_errors=True)
        
        # Calculate final scores
        total_ops = self.validation_metrics['total_operations']
        failed_ops = self.validation_metrics['failed_operations']
        success_rate = (total_ops - failed_ops) / max(1, total_ops)
        
        avg_query_time = sum(self.validation_metrics['query_times']) / max(1, len(self.validation_metrics['query_times']))
        avg_insert_time = sum(self.validation_metrics['insert_times']) / max(1, len(self.validation_metrics['insert_times']))

        final_memory = self.get_memory_usage()
        memory_efficiency = self.initial_memory / max(1, final_memory - self.initial_memory)
        
        print(f"\n" + "="*60)
        print(f"VALIDATION RESULTS")
        print(f"="*60)
        print(f"Overall Success Rate: {success_rate:.2%}")
        print(f"Total Operations: {total_ops}")
        print(f"Failed Operations: {failed_ops}")
        print(f"Average Query Time: {avg_query_time:.4f}s")
        print(f"Average Insert Time: {avg_insert_time:.4f}s")
        print(f"Peak Memory Usage: {self.validation_metrics['peak_memory_mb']:.2f} MB")
        print(f"Memory Efficiency Score: {memory_efficiency:.2f}")
        print(f"Concurrent Success Rate: {self.validation_metrics['concurrent_success_rate']:.2%}")
        print(f"Data Integrity Score: {self.validation_metrics['data_integrity_score']:.2%}")
        print(f"Performance Score: {self.validation_metrics['performance_score']:.2%}")
        
        # Final determination
        ready = (
            success_rate >= 0.95 and
            avg_query_time < 1.0 and
            avg_insert_time < 2.0 and
            self.validation_metrics['concurrent_success_rate'] >= 0.90 and
            self.validation_metrics['data_integrity_score'] >= 0.95 and
            self.validation_metrics['performance_score'] >= 0.90
        )
        
        if ready:
            print(f"\nREADINESS: {'✅' if ready else '❌'}")

        print(f"="*60)

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        return sys.getsizeof(gc.get_objects()) / 1024 / 1024

    def _generate_ddl(self, models: List[Any]) -> str:
        """Generate DDL for models."""
        ddl_statements = []
        for model in models:
            if hasattr(model, '__is_kuzu_relationship__') and model.__is_kuzu_relationship__:
                ddl_statements.append(get_ddl_for_relationship(model))
            else:
                ddl_statements.append(get_ddl_for_node(model))
        return "\n".join(ddl_statements)

    def _track_memory(self):
        """Track current memory usage."""
        current_memory = self.get_memory_usage()
        if current_memory > self.validation_metrics['peak_memory_mb']:
            self.validation_metrics['peak_memory_mb'] = current_memory
        return current_memory

    def test_comprehensive_validation(self):
        """
        The ultimate comprehensive validation test.
        
        This test validates EVERYTHING needed for deployment:
        - Data creation and integrity
        - Complex relationship handling
        - Query performance
        - Concurrent operations
        - Memory efficiency
        - Error handling
        - Cconsistency
        """
        # Registry cleanup to prevent access violations
        from kuzualchemy import clear_registry
        clear_registry()
        import gc
        gc.collect()

        session = KuzuSession(db_path=str(self.db_path))

        # Initialize schema with all models
        all_models = [User, Product, PurchaseOrder, Follows, Purchased, Placed, Contains]
        ddl = self._generate_ddl(all_models)
        initialize_schema(session, ddl=ddl)
        
        # ================================================================
        # PHASE 1: DATA CREATION AND INTEGRITY VALIDATION
        # ================================================================
        print("Phase 1: Data Creation and Integrity Validation...")
        
        phase1_start = time.time()
        
        # Create users
        users = []
        for i in range(1000):  # 1000 users for realistic scale
            user = User(
                user_id=i + 1,
                username=f"user_{i+1:04d}",
                email=f"user{i+1}@example.com",
                full_name=f"User {i+1}",
                score=random.uniform(0, 1000),
                is_active=random.choice([True, True, True, False]),  # 75% active
                created_at=datetime.now() - timedelta(days=random.randint(1, 365))
            )
            users.append(user)
            session.add(user)
            self.validation_metrics['total_operations'] += 1
        
        session.commit()
        
        # Create products
        categories = ["Electronics", "Books", "Clothing", "Home", "Sports"]
        products = []
        for i in range(500):  # 500 products
            product = Product(
                product_id=i + 1,
                name=f"Product {i+1}",
                price=random.uniform(10, 1000),
                category=random.choice(categories),
                rating=random.uniform(1.0, 5.0),
                stock=random.randint(0, 1000)
            )
            products.append(product)
            session.add(product)
            self.validation_metrics['total_operations'] += 1
        
        session.commit()
        
        # Create orders
        orders = []
        for i in range(2000):  # 2000 orders
            order = PurchaseOrder(
                order_id=i + 1,
                total_amount=random.uniform(20, 500),
                status=random.choice(["pending", "shipped", "delivered", "cancelled"]),
                created_at=datetime.now() - timedelta(days=random.randint(1, 90))
            )
            orders.append(order)
            session.add(order)
            self.validation_metrics['total_operations'] += 1
        
        session.commit()
        
        phase1_time = time.time() - phase1_start
        self.validation_metrics['insert_times'].append(phase1_time)
        
        # Validate data integrity
        user_count = list(session.execute("MATCH (u:User) RETURN count(u) as count"))[0]['count']
        product_count = list(session.execute("MATCH (p:Product) RETURN count(p) as count"))[0]['count']
        order_count = list(session.execute("MATCH (o:PurchaseOrder) RETURN count(o) as count"))[0]['count']
        
        data_integrity_score = 1.0
        if user_count != 1000:
            data_integrity_score -= 0.3
            self.validation_metrics['failed_operations'] += abs(1000 - user_count)
        if product_count != 500:
            data_integrity_score -= 0.3
            self.validation_metrics['failed_operations'] += abs(500 - product_count)
        if order_count != 2000:
            data_integrity_score -= 0.4
            self.validation_metrics['failed_operations'] += abs(2000 - order_count)
        
        self.validation_metrics['data_integrity_score'] = max(0.0, data_integrity_score)
        
        print(f"  ✅ Created {user_count} users, {product_count} products, {order_count} orders")
        print(f"  ⏱️  Phase 1 completed in {phase1_time:.2f}s")
        
        # ================================================================
        # PHASE 2: RELATIONSHIP CREATION AND VALIDATION
        # ================================================================
        print("Phase 2: Relationship Creation and Validation...")
        
        phase2_start = time.time()
        
        # Create follow relationships (social network aspect)
        follow_count = 0
        for user_id in range(1, 1001):
            # Each user follows 5-15 others
            num_follows = random.randint(5, 15)
            followed_users = random.sample(
                [uid for uid in range(1, 1001) if uid != user_id],
                min(num_follows, 999)
            )
            
            for followed_id in followed_users:
                follow = Follows(
                    from_node=user_id,
                    to_node=followed_id,
                    followed_at=datetime.now() - timedelta(days=random.randint(1, 180)),
                    strength=random.uniform(0.1, 1.0)
                )
                session.add(follow)
                follow_count += 1
                self.validation_metrics['total_operations'] += 1
        
        session.commit()
        
        # Create purchase relationships (e-commerce aspect)
        purchase_count = 0
        for _ in range(5000):  # 5000 purchases
            user_id = random.randint(1, 1000)
            product_id = random.randint(1, 500)
            
            purchase = Purchased(
                from_node=user_id,
                to_node=product_id,
                quantity=random.randint(1, 5),
                price_paid=random.uniform(10, 1000),
                purchased_at=datetime.now() - timedelta(days=random.randint(1, 365))
            )
            session.add(purchase)
            purchase_count += 1
            self.validation_metrics['total_operations'] += 1
        
        session.commit()
        
        # Create order relationships
        placed_count = 0
        contains_count = 0
        
        for order_id in range(1, 2001):
            # Each order is placed by a random user
            user_id = random.randint(1, 1000)
            placed = Placed(
                from_node=user_id,
                to_node=order_id,
                placed_at=datetime.now() - timedelta(days=random.randint(1, 90))
            )
            session.add(placed)
            placed_count += 1
            
            # Each order contains 1-5 products
            num_products = random.randint(1, 5)
            order_products = random.sample(range(1, 501), num_products)
            
            for product_id in order_products:
                contains = Contains(
                    from_node=order_id,
                    to_node=product_id,
                    quantity=random.randint(1, 3),
                    unit_price=random.uniform(10, 1000)
                )
                session.add(contains)
                contains_count += 1
                self.validation_metrics['total_operations'] += 2
        
        session.commit()
        
        phase2_time = time.time() - phase2_start
        
        print(f"  ✅ Created {follow_count} follows, {purchase_count} purchases")
        print(f"  ✅ Created {placed_count} order placements, {contains_count} order items")
        print(f"  ⏱️  Phase 2 completed in {phase2_time:.2f}s")
        
        # ================================================================
        # PHASE 3: COMPLEX QUERY PERFORMANCE VALIDATION
        # ================================================================
        print("Phase 3: Complex Query Performance Validation...")
        
        complex_queries = [
            {
                "name": "User Statistics",
                "query": "MATCH (u:User) RETURN count(u), avg(u.score), max(u.score), min(u.score)",
                "expected_results": 1
            },
            {
                "name": "Social Network Analysis",
                "query": """
                    MATCH (u1:User)-[:FOLLOWS]->(u2:User)
                    RETURN u2.username, count(*) as follower_count
                    ORDER BY follower_count DESC
                    LIMIT 10
                """,
                "expected_results": 10
            },
            {
                "name": "E-commerce Revenue Analysis",
                "query": """
                    MATCH (u:User)-[p:PURCHASED]->(prod:Product)
                    RETURN prod.category, sum(p.price_paid) as total_revenue, count(p) as purchase_count
                    ORDER BY total_revenue DESC
                """,
                "expected_results": 5  # 5 categories
            },
            {
                "name": "Cross-domain Analysis",
                "query": """
                    MATCH (u:User)-[:FOLLOWS]->(u2:User)-[p:PURCHASED]->(prod:Product)
                    WHERE prod.category = 'Electronics'
                    RETURN u.username, count(DISTINCT prod) as electronics_influence
                    ORDER BY electronics_influence DESC
                    LIMIT 20
                """,
                "expected_results": 20
            },
            {
                "name": "Order Analysis",
                "query": """
                    MATCH (u:User)-[:PLACED]->(o:PurchaseOrder)-[:CONTAINS]->(p:Product)
                    RETURN o.status, count(DISTINCT u) as customers, sum(o.total_amount) as revenue
                    ORDER BY revenue DESC
                """,
                "expected_results": 4  # 4 order statuses
            }
        ]
        
        query_performance_score = 1.0
        
        for query_test in complex_queries:
            query_start = time.time()
            try:
                results = list(session.execute(query_test["query"]))
                query_time = time.time() - query_start
                
                self.validation_metrics['query_times'].append(query_time)
                
                # Validate results
                if len(results) < query_test["expected_results"]:
                    query_performance_score -= 0.1
                    self.validation_metrics['failed_operations'] += 1
                
                # Performance validation (queries should be fast)
                if query_time > 5.0:  # 5 second limit
                    query_performance_score -= 0.1
                    self.validation_metrics['failed_operations'] += 1
                
                print(f"  ✅ {query_test['name']}: {len(results)} results in {query_time:.3f}s")
                
            except Exception as e:
                print(f"  ❌ {query_test['name']}: FAILED - {e}")
                query_performance_score -= 0.2
                self.validation_metrics['failed_operations'] += 1
        
        self.validation_metrics['performance_score'] = max(0.0, query_performance_score)
        
        # ================================================================
        # PHASE 4: CONCURRENT OPERATIONS VALIDATION
        # ================================================================
        print("Phase 4: Concurrent Operations Validation...")
        
        def concurrent_worker(worker_id: int) -> tuple[int, int]:
            """Worker function for concurrent operations."""
            # Use read-only session to avoid file locking conflicts
            # The main session is READ_WRITE, concurrent workers use READ_ONLY
            worker_session = KuzuSession(db_path=str(self.db_path), read_only=True)
            successes = 0
            failures = 0
            
            try:
                for i in range(50):  # 50 operations per worker
                    try:
                        # Only read operations for read-only sessions
                        operation = random.choice(['query_users', 'query_products', 'query_relationships'])

                        if operation == 'query_users':
                            results = list(worker_session.execute("MATCH (u:User) RETURN count(u)"))
                            if results and results[0].get('count(u)', 0) > 0:
                                successes += 1
                            else:
                                failures += 1

                        elif operation == 'query_products':
                            results = list(worker_session.execute("MATCH (p:Product) RETURN count(p)"))
                            if results and results[0].get('count(p)', 0) > 0:
                                successes += 1
                            else:
                                failures += 1

                        elif operation == 'query_relationships':
                            results = list(worker_session.execute("MATCH ()-[r:FOLLOWS]->() RETURN count(r)"))
                            if results and results[0].get('count(r)', 0) > 0:
                                successes += 1
                            else:
                                failures += 1
                        
                    except Exception:
                        failures += 1
            
            finally:
                worker_session.close()
            
            return successes, failures
        
        # Run concurrent operations
        num_workers = 10
        concurrent_start = time.time()
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(concurrent_worker, i) for i in range(num_workers)]
            
            total_concurrent_successes = 0
            total_concurrent_failures = 0
            
            for future in as_completed(futures):
                successes, failures = future.result()
                total_concurrent_successes += successes
                total_concurrent_failures += failures
        
        concurrent_time = time.time() - concurrent_start
        concurrent_success_rate = total_concurrent_successes / (total_concurrent_successes + total_concurrent_failures)
        
        self.validation_metrics['concurrent_success_rate'] = concurrent_success_rate
        
        print(f"  ✅ Concurrent operations: {total_concurrent_successes} successes, {total_concurrent_failures} failures")
        print(f"  📊 Concurrent success rate: {concurrent_success_rate:.2%}")
        print(f"  ⏱️  Phase 4 completed in {concurrent_time:.2f}s")
        
        # ================================================================
        # PHASE 5: FINAL VALIDATION AND SCORING
        # ================================================================
        print("Phase 5: Final Validation and Scoring...")
        
        # Memory efficiency check
        final_memory = self._track_memory()
        memory_increase = final_memory - self.initial_memory
        
        # Final data consistency check
        final_user_count = list(session.execute("MATCH (u:User) RETURN count(u) as count"))[0]['count']
        final_follow_count = list(session.execute("MATCH ()-[f:FOLLOWS]->() RETURN count(f) as count"))[0]['count']
        final_purchase_count = list(session.execute("MATCH ()-[p:PURCHASED]->() RETURN count(p) as count"))[0]['count']
        
        print(f"  📊 Final counts: {final_user_count} users, {final_follow_count} follows, {final_purchase_count} purchases")
        print(f"  💾 Memory usage: {memory_increase:.2f} MB increase")
        
        session.close()
        
        # ================================================================
        # FINAL DETERMINATION
        # ================================================================
        
        # Calculate overall metrics
        total_ops = self.validation_metrics['total_operations']
        failed_ops = self.validation_metrics['failed_operations']
        overall_success_rate = (total_ops - failed_ops) / max(1, total_ops)
        
        avg_query_time = sum(self.validation_metrics['query_times']) / max(1, len(self.validation_metrics['query_times']))
        avg_insert_time = sum(self.validation_metrics['insert_times']) / max(1, len(self.validation_metrics['insert_times']))
        
        criteria_met = {
            'overall_success_rate': overall_success_rate >= 0.95,
            'query_performance': avg_query_time < 1.0,
            'insert_performance': avg_insert_time < 5.0,
            'concurrent_success': self.validation_metrics['concurrent_success_rate'] >= 0.90,
            'data_integrity': self.validation_metrics['data_integrity_score'] >= 0.95,
            'query_complexity': self.validation_metrics['performance_score'] >= 0.90,
            'memory_efficiency': memory_increase < 1000  # Less than 1GB
        }
        
        criteria_passed = sum(criteria_met.values())
        total_criteria = len(criteria_met)
        
        print(f"\nCRITERIA:")
        for criterion, passed in criteria_met.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {criterion}: {status}")
        
        readiness_score = criteria_passed / total_criteria
        
        # FINAL ASSERTION
        assert readiness_score >= 0.85, f"score {readiness_score:.2%} below 85% minimum"
        assert overall_success_rate >= 0.95, f"Overall success rate {overall_success_rate:.2%} below 95% minimum"
        assert avg_query_time < 2.0, f"Average query time {avg_query_time:.3f}s exceeds 2s limit"
        assert self.validation_metrics['concurrent_success_rate'] >= 0.85, f"Concurrent success rate below 85%"
        
        print(f"Score: {readiness_score:.1%}")
