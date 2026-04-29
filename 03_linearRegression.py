import numpy as np
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
# Data 세팅
x_train = np.array([[0.2],[0.8]])
y_train = np.array([[0.3],[0.9]])

# Model 구성
# Input([1]) : x가 1개 (입력), dense(1,) : y가 1개 (출력)
input_layer = Input([1])
output_layer = Dense(1, activation=None)(input_layer)
model = Model(input_layer, output_layer)

# 표 요약
model.summary()
# output_layer 선택
layer = model.layers[1]
# 출력 결과에서 가중치 가져오기
weights = layer.get_weights()

# w (기울기)
w_start = weights[0][0][0]
# b (편향)
b_start = weights[1][0]

print(weights[0])
print(weights[1])
print(w_start, b_start)


from tensorflow.keras.optimizers import Adam, SGD

# Model에 loss 함수와 최적화(learning rate) 지정
model.compile(loss='mean_squared_error', optimizer=SGD(learning_rate = 0.05), metrics=['mse'])
# 학습 (Fitting)
model.fit(x_train, y_train, epochs=30)
ayer = model.layers[1]
# 출력 결과에서 가중치 가져오기
weights = layer.get_weights()

# w (기울기)
w_start = weights[0][0][0]
# b (편향)
b_start = weights[1][0]

print(weights[0])
print(weights[1])
print(w_start, b_start)