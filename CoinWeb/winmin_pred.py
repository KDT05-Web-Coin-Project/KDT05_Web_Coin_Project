import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
 # 가정한 모델 클래스를 불러옵니다.
import torch
import torch.nn as nn

class CryptoPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(CryptoPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # LSTM을 통과시킨 후, 마지막 타임 스텝의 출력만을 사용하여 결과를 예측
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임 스텝의 출력만을 사용
        return out


# 모델 인스턴스화 및 가중치 불러오기
model = CryptoPredictor(input_dim=5, hidden_dim=50, num_layers=2, output_dim=1)
model.load_state_dict(torch.load('model_path.pkl'))  # 모델 가중치 경로
model.eval()

# MinMaxScaler 인스턴스 생성
scaler_features = MinMaxScaler()
scaler_target = MinMaxScaler()

# 예측 함수
def predict(input_df):
    input_features = input_df[['open', 'high', 'low', 'close', 'volume']].values
    input_features_scaled = scaler_features.fit_transform(input_features)
    input_tensor = torch.tensor(input_features_scaled, dtype=torch.float32).unsqueeze(1)  # 배치 차원 추가
    
    scaler_target.fit(pd.read_csv('./DATA/ETHUSDT_1d.csv')[['close']])


    with torch.no_grad():
        predicted_scaled = model(input_tensor)
    predicted_prices = scaler_target.inverse_transform(predicted_scaled.numpy())
    return predicted_prices.flatten()

# 실제 가격과 날짜 추출
def actual(input_df):
    return input_df['close'].values[-10:]

def datetimes(input_df):
    return pd.to_datetime(input_df['timestamp']).dt.strftime('%Y-%m-%d').tolist()