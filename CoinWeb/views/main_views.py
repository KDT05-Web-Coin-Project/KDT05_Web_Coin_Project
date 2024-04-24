from flask import Blueprint, render_template, request, redirect, url_for
from CoinWeb import db

bp = Blueprint('main_views', __name__, url_prefix='/')

@bp.route('/')
def home():
    return render_template('home.html')

# #문섭님에게 접근 : /kms
# @bp.route('/kms')
# def kms():
#     return render_template('kms.html')

# @bp.route('/noo')
# def noo():
#     return render_template('jw_home.html')





# # --------------------------------------------------------------------------------
# # 문섭님 블루프린트 코드
# # bp = ms_bp, url_prefix = '/kms'
# # --------------------------------------------------------------------------------
# ms_bp = Blueprint('main_views', __name__, url_prefix='/kms')

# whattopredict = {
#     'vix': 1,
#     'treasury': 2,
#     'oil': 3,
# }

# @ms_bp.route('/')
# def home():
#     print(predictor.data.shape)
#     result = BTCUSDT_1d.query.order_by(BTCUSDT_1d.timestamp.desc()).limit(100).all()
#     predict = BTCUSDT_1d.query.order_by(BTCUSDT_1d.timestamp.desc()).limit(30).all()
#     result.reverse()
#     predict.reverse()
#     return render_template('test.html', result=result, predict = predict)

# @ms_bp.route('/predict', methods = ['POST'])
# def predict():
#     forms = request.form
#     result = None
#     bitresult = BTCUSDT_1d.query.order_by(BTCUSDT_1d.timestamp.desc()).limit(100).all()
#     bitresult.reverse()
#     bitpredict = BTCUSDT_1d.query.order_by(BTCUSDT_1d.timestamp.desc()).limit(30).all()
#     if forms['symbol'] == 'vix':
#         target = 'VIX'
#         result = Vix.query.order_by(Vix.date.desc()).limit(100).all()
#         predict = predictor.result_vix()
#         result.reverse()
#         predict.reverse()
#         return render_template('result_date.html', result=result, predict=predict, target=target, bitresult = bitresult)
#     if forms['symbol'] == 'coin':
#         target = '코인베이스'
#         result = HistCoin.query.order_by(HistCoin.date.desc()).limit(100).all()
#         predict = predictor.result_coin()
#         result.reverse()
#         predict.reverse()
#         return render_template('result_date.html', result=result, predict=predict, target=target, bitresult = bitresult)
#     if forms['symbol'] == 'oil':
#         target = 'WTI 오일 가격'
#         result = WtiOilPrice.query.order_by(WtiOilPrice.DATE.desc()).limit(100).all()
#         predict = predictor.result_oil()
#         result.reverse()
#         predict.reverse()
#         print(result.__len__())
#         return render_template('result_DATE.html', result=result, predict=predict, target=target, bitresult = bitresult)

#     return render_template('test_alpha.html', result=result, predict=predict)
    
#     # return render_template('test.html', result=result, predict=predict)
#     # result = BTCUSDT_1d.query.order_by(BTCUSDT_1d.timestamp.desc()).limit(100).all()
#     # predict = BTCUSDT_1d.query.order_by(BTCUSDT_1d.timestamp.desc()).limit(30).all()
#     # result.reverse()
#     # predict.reverse()
#     # predict_val = predictor.result_coin()
#     # # return redirect(request.url_root)
#     # return render_template('test_alpha.html', result=result, predict = predict, predict_val = predict_val)
