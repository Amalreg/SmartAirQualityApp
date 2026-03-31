"""
Smart Air Quality Prediction System
Database Models
"""
from database import db
from datetime import datetime

class User(db.Model):
    """
    User model for authentication and session management.
    """
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), default='user')

    def __repr__(self):
        return f"<User {self.email}>"

class SearchHistory(db.Model):
    """
    SearchHistory model tracking every individual user's queries permanently in MySQL.
    """
    __tablename__ = 'search_history'

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    city = db.Column(db.String(100), nullable=False)
    aqi = db.Column(db.Integer, nullable=False)
    predicted_aqi = db.Column(db.Float, nullable=True)
    search_date = db.Column(db.DateTime, default=datetime.utcnow)

    # Automatically map the relationship backwards so User.searches calls all searches gracefully
    user = db.relationship('User', backref=db.backref('searches', lazy=True))
