from flask import Blueprint, render_template, request, redirect, url_for
from CoinWeb import db
from CoinWeb import winmin_pred
import pandas as pd


bp = Blueprint('winmain_main_views', __name__, url_prefix='/win')

@bp.route('/')
def home():
    # 메인 페이지
    return render_template('winmain/winmin_home.html')

@bp.route('/predict', methods=['POST'])
def predict():
    # 데이터 처리 및 예측 로직
    forms = pd.read_csv("./DATA/BTCUSDT_1d_latest.csv")  # 예시로 CSV 파일 사용
    forms_eth = pd.read_csv("./DATA/ETHUSDT_1d.csv")  # 예시로 CSV 파일 사용


    predict = winmin_pred.predict(forms)
    actual = winmin_pred.actual(forms_eth)
    date = winmin_pred.datetimes(forms)
    
    # 예측 결과 페이지 렌더링
    return render_template('winmain/predict.html', input=forms, predict=predict[-10:], actual=actual, labels=date[-10:])