import pandas as pd
import pymysql
from datetime import datetime
from prophet import Prophet
import matplotlib.pyplot as plt 
import math
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine, Connection, text
import koreanize_matplotlib

# 데이터 불러오기 ------------------------------------------------------------------------
engine = create_engine('mysql+pymysql://root:kdt5@1.251.203.204:33065/Team2?charset=utf8mb4')

with engine.begin() as conn:
    data = pd.read_sql(text('SELECT timestamp, close FROM BTCUSDT_1d_latest'), conn)

# dataDF로 변경 --------------------------------------------------------------------
dataDF=data

# data불러오고 컬럼 이름 바꾸기-------------------------------------------------------
dataDF = dataDF.rename(columns={'timestamp':'ds','close' : 'y'})


# test용 생성-----------------------------------------------------------------------
testDF = dataDF.iloc[-7:]
# Prophet---------------------------------------------------------------------------
prophet = Prophet(seasonality_mode='multiplicative',
                  yearly_seasonality=True,
                  weekly_seasonality=True,
                  daily_seasonality=True,
                  changepoint_prior_scale = 0.5)
prophet.fit(dataDF)
# 사용자로부터 periods값 입력받기 
# periods = int(input('periods값을 입력하세요: '))
def predict_bit(periods):
    future_data = prophet.make_future_dataframe(periods=periods, freq='d')
    forecast_data = prophet.predict(future_data)

    # 모델 로드 ---------------------------------------------------------
    import pickle
    import os
    print(os.listdir())
    model_file_path = './prophet_model2.pkl'

    with open(model_file_path, 'wb') as f:
        pickle.dump(prophet,f)

    # 저장된 Prophet 모델 파일을 로드--------------------------------------------------
    with open(model_file_path, 'rb') as f:
        loaded_prophet_model = pickle.load(f)

    # 예측, 테스트, lower,upper --------------------------------------------------------
    pred_y = forecast_data.yhat.values[-periods:]
    test_y = testDF.y.values
    pred_y_lower = forecast_data.yhat_lower.values[-periods:]
    pred_y_upper = forecast_data.yhat_upper.values[-periods:]


    # 시각화 ---------------------------------------------------------------------------
    # rmse = math.sqrt(mean_squared_error(pred_y, test_y))
    plt.plot(pred_y, color="gold", label='예측값') # 예측값
    plt.plot(pred_y_lower, color="red", label = '최소값') # 예측 최소값
    plt.plot(pred_y_upper, color="blue", label='최댓값') # 예측 최댓값
    plt.plot(test_y, color="green", label='실제값') # 실제값
    plt.legend()
    plt.title(f"비트코인 {periods}일 예측 결과")
    plt.savefig('./CoinWeb/static/chart.png', dpi=200)
    plt.close()
    # plt.show()
    # rmse----------------------------------------------------------------------------

if __name__ == '__main__':
    pass
    # predict_bit(7)