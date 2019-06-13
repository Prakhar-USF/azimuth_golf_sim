import os
from flask import Flask
from flask_bootstrap import Bootstrap
from config import Config
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy


def create_app(Config):
    """
    Initialization.
    Create an application instance which handles all requests.

    :param Config: config file
    """
    application = Flask(__name__)
    application.config.from_object(Config)
    return application


application = create_app(Config)
db = SQLAlchemy(application)

with application.app_context():
    db.init_app(application)
    db.create_all()
    db.session.commit()

try:
    os.makedirs(application.instance_path)
except OSError:
    pass

# login_manager needs to be initiated before running the app
login_manager = LoginManager()
login_manager.init_app(application)

bootstrap = Bootstrap(application)

# Added at the bottom to avoid circular dependencies
# Altough it violates PEP8 standards
from app import db_module
from app import routes



