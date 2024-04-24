from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()


## Application Factory 함수 ---------
def create_app():
    # Flask 인스턴스 생성
    app = Flask(__name__)

    # 설정 내용 로딩
    app.config.from_pyfile("config.py")

    # ORM 즉, DB 초기화
    db.init_app(app)
    migrate.init_app(app, db)

    # 테이블 클래스
    from . import model

    # 블루프린트
    from .views import main_views, ms_main_views, yl_main_views, jw_main_views, winmin_main_views

    app.register_blueprint(main_views.bp)
    app.register_blueprint(ms_main_views.ms_bp)
    app.register_blueprint(yl_main_views.yl_bp)
    app.register_blueprint(winmin_main_views.bp)
    app.register_blueprint(jw_main_views.bp)


    # Flask Server 인스턴스 반환
    return app
