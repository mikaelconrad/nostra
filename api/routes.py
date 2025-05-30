"""
API route definitions and organization
"""

from flask import Blueprint

# Create blueprints for different API sections
crypto_bp = Blueprint('crypto', __name__, url_prefix='/api/crypto')
portfolio_bp = Blueprint('portfolio', __name__, url_prefix='/api/portfolio')
analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')
admin_bp = Blueprint('admin', __name__, url_prefix='/api/admin')

def register_blueprints(app):
    """Register all blueprints with the Flask app"""
    app.register_blueprint(crypto_bp)
    app.register_blueprint(portfolio_bp)
    app.register_blueprint(analytics_bp)
    app.register_blueprint(admin_bp)