from sqlalchemy import Column, String, Boolean, Date, Float, BigInteger, Integer, ForeignKey, create_engine, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

Base = declarative_base()

class Ticker(Base):
    """
    Stores basic information for each company.
    """
    __tablename__ = 'tickers'
    ticker = Column(String(20), primary_key=True)
    company_name = Column(String(255))
    sector = Column(String(255))
    is_delisted = Column(Boolean, default=False)
    
    # Relationships
    memberships = relationship("IndexMembership", back_populates="ticker_ref")
    prices = relationship("DailyPrice", back_populates="ticker_ref")


class IndexMembership(Base):
    """
    Records the date intervals when each stock was part of the S&P 500.
    """
    __tablename__ = 'index_memberships'
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), ForeignKey('tickers.ticker'))
    start_date = Column(Date, nullable=False)
    # end_date is None if the stock is currently in the S&P 500
    end_date = Column(Date, nullable=True) 

    # Relationship
    ticker_ref = relationship("Ticker", back_populates="memberships")


class DailyPrice(Base):
    """
    Daily price history for each stock.
    """
    __tablename__ = 'daily_prices'
    __table_args__ = (UniqueConstraint('ticker', 'date', name='uq_ticker_date'),)
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), ForeignKey('tickers.ticker'))
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    adj_close = Column(Float)
    volume = Column(BigInteger)

    # Relationship
    ticker_ref = relationship("Ticker", back_populates="prices")


# Database Setup functions
def get_engine(db_path='sqlite:///sp500_data.db'):
    return create_engine(db_path)

def init_db(engine):
    """Creates the database tables based on models."""
    Base.metadata.create_all(bind=engine)

def get_session(engine):
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    return SessionLocal()

if __name__ == "__main__":
    engine = get_engine()
    init_db(engine)
    print("Database models initialized and tables created.")
