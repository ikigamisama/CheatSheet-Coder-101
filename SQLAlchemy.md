# SQLAlchemy Cheat Sheet

## Basic Setup & Connection

```python
from sqlalchemy import create_engine, Column, Integer, String, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Create engine
engine = create_engine('sqlite:///example.db')  # SQLite
# engine = create_engine('postgresql://user:pass@localhost/dbname')  # PostgreSQL
# engine = create_engine('mysql://user:pass@localhost/dbname')  # MySQL

# Create base class
Base = declarative_base()

# Create session
Session = sessionmaker(bind=engine)
session = Session()
```

## Model Definition

```python
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), nullable=False)
    age = Column(Integer)

    # Relationship
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = 'posts'

    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    content = Column(String(500))
    user_id = Column(Integer, ForeignKey('users.id'))

    # Relationship
    author = relationship("User", back_populates="posts")

# Create tables
Base.metadata.create_all(engine)
```

## Common Column Types

```python
from sqlalchemy import Boolean, DateTime, Text, Float, Numeric
from datetime import datetime

class Example(Base):
    __tablename__ = 'examples'

    id = Column(Integer, primary_key=True)
    name = Column(String(50))           # VARCHAR
    description = Column(Text)          # TEXT
    price = Column(Float)               # FLOAT
    precise_price = Column(Numeric(10, 2))  # DECIMAL
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

## CRUD Operations

### Create (Insert)

```python
# Single record
user = User(username='john_doe', email='john@example.com', age=25)
session.add(user)
session.commit()

# Multiple records
users = [
    User(username='alice', email='alice@example.com'),
    User(username='bob', email='bob@example.com')
]
session.add_all(users)
session.commit()
```

### Read (Query)

```python
# Get all records
users = session.query(User).all()

# Get first record
user = session.query(User).first()

# Get by primary key
user = session.query(User).get(1)

# Filter
users = session.query(User).filter(User.age > 18).all()
user = session.query(User).filter(User.username == 'john_doe').first()

# Multiple filters
users = session.query(User).filter(User.age > 18, User.email.like('%@gmail.com')).all()

# Filter with AND/OR
from sqlalchemy import and_, or_
users = session.query(User).filter(
    and_(User.age > 18, User.username.like('j%'))
).all()

users = session.query(User).filter(
    or_(User.age > 65, User.age < 18)
).all()
```

### Update

```python
# Update single record
user = session.query(User).filter(User.username == 'john_doe').first()
user.age = 26
session.commit()

# Bulk update
session.query(User).filter(User.age < 18).update({User.age: 18})
session.commit()
```

### Delete

```python
# Delete single record
user = session.query(User).filter(User.username == 'john_doe').first()
session.delete(user)
session.commit()

# Bulk delete
session.query(User).filter(User.age < 18).delete()
session.commit()
```

## Query Methods & Filters

```python
# Ordering
users = session.query(User).order_by(User.age).all()
users = session.query(User).order_by(User.age.desc()).all()

# Limiting
users = session.query(User).limit(5).all()
users = session.query(User).offset(10).limit(5).all()

# Counting
count = session.query(User).count()

# Distinct
usernames = session.query(User.username).distinct().all()

# Filter operators
users = session.query(User).filter(User.age.in_([25, 30, 35])).all()
users = session.query(User).filter(User.username.like('%john%')).all()
users = session.query(User).filter(User.age.between(18, 65)).all()
users = session.query(User).filter(User.email.isnull()).all()
users = session.query(User).filter(User.email.isnot(None)).all()
```

## Relationships & Joins

```python
# One-to-Many relationship query
user = session.query(User).filter(User.username == 'john_doe').first()
user_posts = user.posts  # Access related posts

# Join queries
results = session.query(User, Post).join(Post).all()

# Left join
results = session.query(User).outerjoin(Post).all()

# Filter on related table
users_with_posts = session.query(User).join(Post).filter(
    Post.title.like('%Python%')
).all()
```

## Relationship Types

```python
# One-to-Many
class Department(Base):
    __tablename__ = 'departments'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    employees = relationship("Employee", back_populates="department")

class Employee(Base):
    __tablename__ = 'employees'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    department_id = Column(Integer, ForeignKey('departments.id'))
    department = relationship("Department", back_populates="employees")

# Many-to-Many
from sqlalchemy import Table

user_role_association = Table(
    'user_roles', Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id')),
    Column('role_id', Integer, ForeignKey('roles.id'))
)

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String(50))
    roles = relationship("Role", secondary=user_role_association, back_populates="users")

class Role(Base):
    __tablename__ = 'roles'
    id = Column(Integer, primary_key=True)
    name = Column(String(50))
    users = relationship("User", secondary=user_role_association, back_populates="roles")
```

## Aggregation & Grouping

```python
from sqlalchemy import func

# Count
user_count = session.query(func.count(User.id)).scalar()

# Group by
age_counts = session.query(User.age, func.count(User.id)).group_by(User.age).all()

# Having
popular_ages = session.query(User.age, func.count(User.id)).group_by(User.age).having(func.count(User.id) > 1).all()

# Min, Max, Avg, Sum
stats = session.query(
    func.min(User.age),
    func.max(User.age),
    func.avg(User.age),
    func.sum(User.age)
).first()
```

## Raw SQL

```python
# Execute raw SQL
result = session.execute("SELECT * FROM users WHERE age > :age", {"age": 18})
users = result.fetchall()

# Using text()
from sqlalchemy import text
result = session.execute(text("SELECT * FROM users WHERE age > :age"), {"age": 18})
```

## Session Management

```python
# Context manager (recommended)
with Session() as session:
    user = User(username='test')
    session.add(user)
    session.commit()

# Manual session management
session = Session()
try:
    user = User(username='test')
    session.add(user)
    session.commit()
except Exception:
    session.rollback()
    raise
finally:
    session.close()
```

## Common Patterns

```python
# Get or create
def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance, False
    else:
        instance = model(**kwargs)
        session.add(instance)
        return instance, True

user, created = get_or_create(session, User, username='john')

# Pagination helper
def paginate(query, page, per_page=20):
    return query.offset((page - 1) * per_page).limit(per_page)

users = paginate(session.query(User), page=1, per_page=10).all()
```

## Best Practices

1. **Always use sessions in context managers** when possible
2. **Commit explicitly** after making changes
3. **Handle exceptions** and rollback on errors
4. **Close sessions** when done (automatic with context managers)
5. **Use relationships** instead of manual joins when possible
6. **Index frequently queried columns**
7. **Use bulk operations** for large datasets
8. **Lazy load relationships** by default, eager load when needed

## Quick Reference Commands

```bash
# Install SQLAlchemy
pip install sqlalchemy

# With database drivers
pip install sqlalchemy psycopg2-binary  # PostgreSQL
pip install sqlalchemy pymysql          # MySQL
```
