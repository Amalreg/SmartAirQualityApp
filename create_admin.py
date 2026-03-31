from app import app
from database import db
from models import User
from werkzeug.security import generate_password_hash

def create_admin():
    with app.app_context():
        # Check if admin already exists
        admin_email = "admin@example.com"
        admin = User.query.filter_by(email=admin_email).first()
        
        if not admin:
            password_hash = generate_password_hash("admin123")
            new_admin = User(
                name="Admin User",
                email=admin_email,
                password_hash=password_hash,
                role="admin"
            )
            db.session.add(new_admin)
            db.session.commit()
            print(f"Admin user created successfully: {admin_email}")
        else:
            # Update existing admin password and role if needed
            admin.role = "admin"
            admin.password_hash = generate_password_hash("admin123")
            db.session.commit()
            print(f"Admin user already exists. Role and password updated for: {admin_email}")

if __name__ == "__main__":
    create_admin()
