from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

def sqlalchemy_to_dict(obj):
    if isinstance(obj, Base):
        return {column.name: getattr(obj, column.name) for column in obj.__table__.columns}
    raise TypeError(f"Expected a SQLAlchemy Base instance, got {type(obj)} instead.")

def to_plain_object(obj):
    # 创建一个新的对象
    plain_object = type('PlainObject', (), {})()

    # 将 SQLAlchemy 模型实例的属性复制到新对象
    for column in obj.__table__.columns:
        setattr(plain_object, column.name, getattr(obj, column.name))

    return plain_object

class PgClient():
    conn_params = {}
    def __init__(self,dbname,user,password,host,port):
        self.conn_params["dbname"] = dbname
        self.conn_params["user"] = user
        self.conn_params["password"] = password
        self.conn_params["host"] = host
        self.conn_params["port"] = port
        self.database_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{dbname}"
        self.engine = create_engine(self.database_url)

    @contextmanager
    def get_session(self):
        Session = sessionmaker(bind=self.engine)
        session = Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            raise ValueError(f"错误回滚{e}")
        finally:
            session.close()