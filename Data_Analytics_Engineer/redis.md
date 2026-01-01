# Redis Cheat Sheet & Production Code Guide

## Core Data Types

### 1. String

**Commands:**

- `SET key value` - Set a key
- `GET key` - Get a key
- `SETEX key seconds value` - Set with expiration
- `INCR key` - Increment by 1
- `DECR key` - Decrement by 1
- `INCRBY key amount` - Increment by amount

**Use Cases:**

- **Session Management** - Store user session data
- **Caching** - Cache API responses, database queries
- **Distributed Lock** - Implement mutex with `SET key value NX EX seconds`

### 2. Hash

**Commands:**

- `HSET key field value` - Set hash field
- `HGET key field` - Get hash field
- `HGETALL key` - Get all fields
- `HMSET key field1 value1 field2 value2` - Set multiple fields
- `HINCRBY key field amount` - Increment field

**Use Cases:**

- **Shopping Cart** - Store cart items with quantities
- **User Profile** - Store user attributes
- **Configuration** - Store app settings

### 3. List

**Commands:**

- `LPUSH key value` - Push to left
- `RPUSH key value` - Push to right
- `LPOP key` - Pop from left
- `RPOP key` - Pop from right
- `LRANGE key start stop` - Get range
- `BLPOP key timeout` - Blocking pop

**Use Cases:**

- **Message Queue** - Task queue, job processing
- **Activity Feed** - Recent activities
- **Timeline** - Social media feeds

### 4. Set

**Commands:**

- `SADD key member` - Add member
- `SMEMBERS key` - Get all members
- `SISMEMBER key member` - Check membership
- `SINTER key1 key2` - Intersection
- `SUNION key1 key2` - Union

**Use Cases:**

- **Tags** - Unique tag collections
- **Followers** - Social network relationships
- **IP Whitelist/Blacklist**

### 5. Sorted Set (ZSet)

**Commands:**

- `ZADD key score member` - Add with score
- `ZRANGE key start stop [WITHSCORES]` - Get range (ascending)
- `ZREVRANGE key start stop` - Get range (descending)
- `ZINCRBY key amount member` - Increment score
- `ZRANK key member` - Get rank (0-based)

**Use Cases:**

- **Leaderboard** - Gaming scores, rankings
- **Priority Queue** - Task scheduling
- **Rate Limiter** - Sliding window

### 6. Bitmap

**Commands:**

- `SETBIT key offset value` - Set bit
- `GETBIT key offset value` - Get bit
- `BITCOUNT key` - Count set bits
- `BITOP operation destkey key1 key2` - Bitwise operations

**Use Cases:**

- **User Retention** - Daily active users
- **Real-time Analytics** - User attendance tracking
- **Feature Flags** - Per-user feature toggles

## Common Patterns

### Time-to-Live (TTL)

```redis
EXPIRE key seconds
TTL key
PERSIST key
```

### Transactions

```redis
MULTI
SET key1 value1
SET key2 value2
EXEC
```

### Pub/Sub

```redis
SUBSCRIBE channel
PUBLISH channel message
```

## Top Production Use Cases

1. **Session Management** - Fast user session storage with automatic expiration
2. **Caching** - Reduce database load with intelligent caching layer
3. **Rate Limiting** - API throttling, request limiting
4. **Distributed Locking** - Coordinate distributed systems
5. **Leaderboards** - Real-time rankings and scoring
6. **Message Queues** - Asynchronous task processing
7. **Real-time Analytics** - Count unique visitors, track events

## Performance Tips

- Use pipelining for bulk operations
- Set appropriate TTL to prevent memory bloat
- Use `SCAN` instead of `KEYS` in production
- Monitor memory usage with `INFO memory`
- Use connection pooling
- Enable persistence (RDB/AOF) for data durability

---

# Python Production Code Examples

## Installation

```bash
pip install redis
```

## 1. Session Management

**Definition:** Session management stores user authentication and state data in Redis, allowing fast access across multiple servers without database queries.

**Purpose:**

- Maintain user login state across distributed applications
- Enable horizontal scaling without sticky sessions
- Provide fast session lookups (sub-millisecond)
- Automatic session expiration for security
- Share sessions across microservices

```python
import redis
import json
import uuid
from datetime import datetime

class SessionManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def create_session(self, user_id: str, data: dict, ttl: int = 3600) -> str:
        """Create user session with auto-expiration"""
        session_id = str(uuid.uuid4())
        key = f"session:{session_id}"

        data['user_id'] = user_id
        data['created_at'] = datetime.now().isoformat()

        self.client.setex(key, ttl, json.dumps(data))
        return session_id

    def get_session(self, session_id: str) -> dict:
        """Retrieve session data"""
        key = f"session:{session_id}"
        data = self.client.get(key)
        return json.loads(data) if data else None

    def delete_session(self, session_id: str):
        """Delete session (logout)"""
        self.client.delete(f"session:{session_id}")

# Usage
sm = SessionManager()
session_id = sm.create_session('user123', {'name': 'John', 'role': 'admin'})
session_data = sm.get_session(session_id)
```

## 2. Caching Layer

**Definition:** A caching layer stores frequently accessed data in memory to reduce database load and improve response times by orders of magnitude.

**Purpose:**

- Reduce database query load by 70-90%
- Dramatically improve API response times (from seconds to milliseconds)
- Lower infrastructure costs by reducing database reads
- Handle traffic spikes without overwhelming backend systems
- Cache expensive computations and API responses
- Implement cache invalidation strategies

```python
import redis
import json
import hashlib
from functools import wraps

class CacheManager:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        data = json.dumps({'args': args, 'kwargs': kwargs}, sort_keys=True)
        hash_key = hashlib.md5(data.encode()).hexdigest()
        return f"{prefix}:{hash_key}"

    def get(self, key: str):
        """Get cached value"""
        data = self.client.get(key)
        return json.loads(data) if data else None

    def set(self, key: str, value, ttl: int = 300):
        """Set cache with TTL"""
        self.client.setex(key, ttl, json.dumps(value))

    def delete(self, key: str):
        """Delete cache"""
        self.client.delete(key)

def cached(prefix: str, ttl: int = 300):
    """Decorator for caching function results"""
    def decorator(func):
        cache = CacheManager()

        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache.cache_key(prefix, *args, **kwargs)
            result = cache.get(cache_key)

            if result is None:
                result = func(*args, **kwargs)
                cache.set(cache_key, result, ttl)

            return result
        return wrapper
    return decorator

# Usage
@cached('user_profile', ttl=600)
def get_user_profile(user_id):
    # Expensive database query
    return {'id': user_id, 'name': 'John Doe'}

profile = get_user_profile(123)
```

## 3. Rate Limiter

**Definition:** Rate limiting controls how many requests a user or service can make within a time window, preventing abuse and ensuring fair resource usage.

**Purpose:**

- Protect APIs from abuse and DDoS attacks
- Ensure fair usage across all users
- Prevent service degradation during traffic spikes
- Implement tiered access (free vs paid users)
- Comply with third-party API rate limits
- Reduce costs from excessive API calls
- Maintain system stability under load

```python
import redis
import time

class RateLimiter:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def is_allowed(self, identifier: str, max_requests: int = 100,
                    window: int = 60) -> bool:
        """
        Sliding window rate limiter
        identifier: user_id, ip_address, api_key
        max_requests: maximum requests allowed
        window: time window in seconds
        """
        key = f"rate_limit:{identifier}"
        now = time.time()

        pipe = self.client.pipeline()
        pipe.zremrangebyscore(key, 0, now - window)
        pipe.zcard(key)
        pipe.zadd(key, {str(now): now})
        pipe.expire(key, window)

        results = pipe.execute()
        current_requests = results[1]

        return current_requests < max_requests

    def get_remaining(self, identifier: str, max_requests: int = 100,
                      window: int = 60) -> int:
        """Get remaining requests"""
        key = f"rate_limit:{identifier}"
        now = time.time()

        self.client.zremrangebyscore(key, 0, now - window)
        current = self.client.zcard(key)
        return max(0, max_requests - current)

# Usage
limiter = RateLimiter()

def api_endpoint(user_id):
    if not limiter.is_allowed(user_id, max_requests=100, window=60):
        return {'error': 'Rate limit exceeded'}, 429

    # Process request
    return {'success': True}, 200
```

## 4. Distributed Lock

**Definition:** A distributed lock ensures that only one process can execute a critical section of code at a time across multiple servers, preventing race conditions and data corruption.

**Purpose:**

- Prevent duplicate payment processing
- Ensure only one worker processes a job
- Coordinate inventory updates across servers
- Prevent race conditions in distributed systems
- Implement leader election patterns
- Synchronize file uploads or data imports
- Maintain data consistency in microservices

```python
import redis
import uuid
import time

class DistributedLock:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(host=host, port=port, db=db)

    def acquire(self, lock_name: str, timeout: int = 10,
                retry_interval: float = 0.1) -> str:
        """Acquire distributed lock"""
        identifier = str(uuid.uuid4())
        lock_key = f"lock:{lock_name}"
        end_time = time.time() + timeout

        while time.time() < end_time:
            if self.client.set(lock_key, identifier, nx=True, ex=timeout):
                return identifier
            time.sleep(retry_interval)

        return None

    def release(self, lock_name: str, identifier: str) -> bool:
        """Release distributed lock"""
        lock_key = f"lock:{lock_name}"

        # Lua script for atomic check and delete
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """

        return self.client.eval(lua_script, 1, lock_key, identifier) == 1

# Usage with context manager
from contextlib import contextmanager

@contextmanager
def distributed_lock(lock_name: str, timeout: int = 10):
    lock = DistributedLock()
    identifier = lock.acquire(lock_name, timeout)

    if identifier is None:
        raise Exception(f"Could not acquire lock: {lock_name}")

    try:
        yield
    finally:
        lock.release(lock_name, identifier)

# Usage
with distributed_lock('process_payment'):
    # Critical section - only one process executes this
    process_payment()
```

## 5. Gaming Leaderboard

**Definition:** A leaderboard system ranks players by score in real-time using Redis sorted sets, providing instant updates and queries for competitive gaming features.

**Purpose:**

- Display real-time rankings for competitive games
- Enable social features (comparing with friends)
- Motivate user engagement through competition
- Support seasonal/tournament leaderboards
- Provide instant score updates without database lag
- Scale to millions of players efficiently
- Power achievement and reward systems

```python
import redis
from typing import List, Tuple

class Leaderboard:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def add_score(self, leaderboard: str, player: str, score: float):
        """Add or update player score"""
        key = f"leaderboard:{leaderboard}"
        self.client.zadd(key, {player: score})

    def increment_score(self, leaderboard: str, player: str, amount: float):
        """Increment player score"""
        key = f"leaderboard:{leaderboard}"
        self.client.zincrby(key, amount, player)

    def get_top(self, leaderboard: str, count: int = 10) -> List[Tuple[str, float]]:
        """Get top N players"""
        key = f"leaderboard:{leaderboard}"
        return self.client.zrevrange(key, 0, count - 1, withscores=True)

    def get_rank(self, leaderboard: str, player: str) -> int:
        """Get player rank (1-based)"""
        key = f"leaderboard:{leaderboard}"
        rank = self.client.zrevrank(key, player)
        return rank + 1 if rank is not None else None

    def get_score(self, leaderboard: str, player: str) -> float:
        """Get player score"""
        key = f"leaderboard:{leaderboard}"
        return self.client.zscore(key, player)

    def get_around_player(self, leaderboard: str, player: str,
                          range_size: int = 5) -> List[Tuple[str, float]]:
        """Get players around a specific player"""
        key = f"leaderboard:{leaderboard}"
        rank = self.client.zrevrank(key, player)

        if rank is None:
            return []

        start = max(0, rank - range_size)
        end = rank + range_size

        return self.client.zrevrange(key, start, end, withscores=True)

# Usage
lb = Leaderboard()

# Add scores
lb.add_score('global', 'player1', 1500)
lb.add_score('global', 'player2', 2000)
lb.increment_score('global', 'player1', 50)

# Get rankings
top_players = lb.get_top('global', 10)
rank = lb.get_rank('global', 'player1')
nearby = lb.get_around_player('global', 'player1', range_size=3)
```

## 6. Shopping Cart (Hash)

**Definition:** A shopping cart stores user's selected items using Redis hashes, providing fast access and updates without database writes for every cart modification.

**Purpose:**

- Provide instant cart updates without page refresh
- Persist carts across sessions and devices
- Reduce database load from frequent cart changes
- Support guest checkout (cart before login)
- Enable cart recovery for abandoned checkouts
- Scale during high-traffic sales events
- Store temporary data with automatic expiration

```python
import redis
import json

class ShoppingCart:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def add_item(self, user_id: str, product_id: str, quantity: int,
                 product_data: dict):
        """Add item to cart"""
        cart_key = f"cart:{user_id}"
        item_data = {
            'quantity': quantity,
            'product': product_data
        }
        self.client.hset(cart_key, product_id, json.dumps(item_data))
        self.client.expire(cart_key, 86400 * 7)  # 7 days TTL

    def remove_item(self, user_id: str, product_id: str):
        """Remove item from cart"""
        cart_key = f"cart:{user_id}"
        self.client.hdel(cart_key, product_id)

    def update_quantity(self, user_id: str, product_id: str, quantity: int):
        """Update item quantity"""
        cart_key = f"cart:{user_id}"
        item_data = self.client.hget(cart_key, product_id)

        if item_data:
            data = json.loads(item_data)
            data['quantity'] = quantity
            self.client.hset(cart_key, product_id, json.dumps(data))

    def get_cart(self, user_id: str) -> dict:
        """Get all cart items"""
        cart_key = f"cart:{user_id}"
        cart = self.client.hgetall(cart_key)
        return {k: json.loads(v) for k, v in cart.items()}

    def clear_cart(self, user_id: str):
        """Clear cart"""
        self.client.delete(f"cart:{user_id}")

    def get_total(self, user_id: str) -> float:
        """Calculate cart total"""
        cart = self.get_cart(user_id)
        total = sum(
            item['quantity'] * item['product']['price']
            for item in cart.values()
        )
        return total

# Usage
cart = ShoppingCart()
cart.add_item('user123', 'prod1', 2, {'name': 'Laptop', 'price': 999.99})
cart.add_item('user123', 'prod2', 1, {'name': 'Mouse', 'price': 29.99})
total = cart.get_total('user123')
```

## 7. Message Queue (List)

**Definition:** A message queue enables asynchronous task processing by storing jobs in Redis lists, allowing workers to process tasks independently without blocking the main application.

**Purpose:**

- Offload slow operations (email sending, image processing)
- Decouple services for better scalability
- Handle background jobs without blocking users
- Ensure tasks are processed even if servers restart
- Enable retry logic for failed tasks
- Distribute work across multiple workers
- Implement priority queues for urgent tasks

```python
import redis
import json
import time
from typing import Optional

class MessageQueue:
    def __init__(self, host='localhost', port=6379, db=0):
        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )

    def enqueue(self, queue_name: str, message: dict):
        """Add message to queue"""
        queue_key = f"queue:{queue_name}"
        self.client.rpush(queue_key, json.dumps(message))

    def dequeue(self, queue_name: str, timeout: int = 0) -> Optional[dict]:
        """Remove and return message from queue"""
        queue_key = f"queue:{queue_name}"

        if timeout > 0:
            result = self.client.blpop(queue_key, timeout)
            return json.loads(result[1]) if result else None
        else:
            result = self.client.lpop(queue_key)
            return json.loads(result) if result else None

    def size(self, queue_name: str) -> int:
        """Get queue size"""
        return self.client.llen(f"queue:{queue_name}")

    def peek(self, queue_name: str, count: int = 1) -> list:
        """View messages without removing"""
        queue_key = f"queue:{queue_name}"
        messages = self.client.lrange(queue_key, 0, count - 1)
        return [json.loads(msg) for msg in messages]

# Worker example
def worker(queue_name: str):
    mq = MessageQueue()

    while True:
        message = mq.dequeue(queue_name, timeout=5)

        if message:
            try:
                # Process message
                print(f"Processing: {message}")
                # Do work here
                time.sleep(1)
            except Exception as e:
                print(f"Error processing message: {e}")
                # Re-queue or move to dead letter queue
        else:
            print("No messages, waiting...")

# Producer
mq = MessageQueue()
mq.enqueue('tasks', {'task': 'send_email', 'to': 'user@example.com'})
```

## Complete Production Setup

```python
import redis
from redis.sentinel import Sentinel
from redis.cluster import RedisCluster

class RedisFactory:
    @staticmethod
    def create_standalone(host='localhost', port=6379, db=0):
        """Create standalone Redis connection"""
        return redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_keepalive=True,
            socket_connect_timeout=5,
            retry_on_timeout=True,
            health_check_interval=30
        )

    @staticmethod
    def create_sentinel(sentinels, service_name, db=0):
        """Create Redis Sentinel connection for high availability"""
        sentinel = Sentinel(
            sentinels,
            socket_timeout=0.5,
            db=db
        )
        return sentinel.master_for(
            service_name,
            decode_responses=True,
            socket_keepalive=True
        )

    @staticmethod
    def create_cluster(startup_nodes):
        """Create Redis Cluster connection"""
        return RedisCluster(
            startup_nodes=startup_nodes,
            decode_responses=True,
            skip_full_coverage_check=True,
            max_connections_per_node=50
        )

# Production usage
client = RedisFactory.create_standalone(
    host='redis.production.com',
    port=6379
)
```
