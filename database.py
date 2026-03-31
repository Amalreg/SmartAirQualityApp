"""
Smart Air Quality Prediction System
Database Connection File
"""

# Example using SQLAlchemy, modify according to your specific database choice (e.g., PyMongo, psycopg2)

from flask_sqlalchemy import SQLAlchemy
import os

db = SQLAlchemy()

def init_db(app):
    """
    Initialize the database connection with the Flask app.
    
    Args:
        app (Flask): The Flask application instance.
    """
    # Configure database URI from environment variable
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    db.init_app(app)
    
    # Create tables if they don't exist
    with app.app_context():
        # Import models here to avoid circular imports
        from models import User, SearchHistory
        db.create_all()

def get_session():
    """
    Helper function to get a database session.
    """
    return db.session
