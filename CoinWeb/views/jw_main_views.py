from flask import Blueprint, render_template, request, redirect, url_for
from CoinWeb import db
from CoinWeb import noo_pred
import pandas as pd

bp = Blueprint('jw_main_views', __name__, url_prefix='/noo')

@bp.route('/')
def home():
    return render_template('noo/jw_home.html')

@bp.route('/input', methods=['POST'])
def input():
    # forms = request.form
    forms = pd.read_csv("./DATA/data_with_btc_scaled.csv")
    forms = noo_pred.make_data(forms)
    predict = noo_pred.predict(forms)
    actual = noo_pred.actual(forms)
    date = noo_pred.datetimes(forms)
    return render_template('noo/predict.html', input = forms, predict = predict, actual = actual, labels = date)