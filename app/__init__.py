from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False
    from app.routes import main
    app.register_blueprint(main)
    return app
