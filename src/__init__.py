import os
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_cors import CORS
from dotenv import load_dotenv
from datetime import datetime, timezone

# Load environment variables
load_dotenv()

# Initialize database
db = SQLAlchemy()

class Run(db.Model):
    __tablename__ = 'runs'
    id = db.Column(db.Integer, primary_key=True)
    status = db.Column(db.String(50), nullable=True)
    podcast_id = db.Column(db.String(255), nullable=True)
    fired = db.Column(db.Boolean, default=False)
    
    # New fields
    user_id = db.Column(db.Integer, nullable=True)
    model_name = db.Column(db.String(255), nullable=True)
    model_type = db.Column(db.String(255), nullable=True)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, nullable=True, default=lambda: datetime.now(tz=timezone.utc))
    updated_at = db.Column(db.DateTime, nullable=True, default=lambda: datetime.now(tz=timezone.utc), onupdate=lambda: datetime.now(tz=timezone.utc))

class User(db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    name = db.Column(db.String(255))
    picture = db.Column(db.String(255))
    run_id = db.Column(db.Integer, db.ForeignKey('runs.id'), nullable=True)

    run = db.relationship('Run', backref='user', lazy='joined', uselist=False)

def create_app():
    # Initialize Flask app
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    # Load configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize database and migration tools
    db.init_app(app)
    Migrate(app, db)

    return app