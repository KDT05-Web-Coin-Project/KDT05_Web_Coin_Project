from flask import Blueprint, render_template, request, redirect, url_for
from CoinWeb import db
from CoinWeb import predictor
from CoinWeb import coinSYL


yl_bp = Blueprint('yl_main_views', __name__, url_prefix='/syl')


@yl_bp.route('/')
def home():
    return render_template('ylindex.html', path = '')

@yl_bp.route('/predict', methods = ['POST'])
def predict():
    formdata = request.form
    periods = formdata['periods']
    coinSYL.predict_bit(int(periods))
    return render_template('ylindex.html', path = 'exist')
    