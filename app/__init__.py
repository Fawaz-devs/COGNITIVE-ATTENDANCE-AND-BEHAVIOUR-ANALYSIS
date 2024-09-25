from flask import Flask
from flask_login import LoginManager
from config.settings import Config

app = Flask(__name__)
app.config.from_object(Config)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'

from app.routes import auth, main, stats

app.register_blueprint(auth.bp)
app.register_blueprint(main.bp)
app.register_blueprint(stats.bp)

from app.models.user import User

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))