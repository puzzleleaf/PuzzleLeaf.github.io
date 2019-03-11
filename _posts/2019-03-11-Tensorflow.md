---
title: "Tensorflow Free Course"
categories: ML
tags: ML
---

Udacity의 [텐서플로우 무료 강좌](https://classroom.udacity.com/courses/ud187)가 있어서 공부하면서 정리를 해보려고 한다.

짧은 동영상으로 배경 설명을 하고, 주로 CodeLab에서 직접 실습하면서 배우는 방식으로 강의가 진행된다.

무료 강의는 총 5개로 이루어져 있다.


**Lessen 01** : [CodeLab](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb#scrollTo=7b5jv0ouFREV)

파이썬의 기초와 Numpy 사용방법이 주가 되는 내용이다.

#### Numpy and lists

```python
import numpy as np

a = np.array(["hello", "world"])
a = np.append(a, "!")
print("현재 배열: {}", format(a))

for i in a:
  print(i)

for i,e in enumerate(a):
  print("Index: {}, was: {}".format(i, e))

#현재 배열: {} ['hello' 'world' '!']
# hello
# world
# !
# Index: 0, was: hello
# Index: 1, was: world
# Index: 2, was: !

print("배열의 기본 연산")
b = np.array([0,1,4,3,2])
print("Max: {}".format(np.max(b)))
print("Average: {}".format(np.average(b)))
print("Max index: {}".format(np.argmax(b)))

# 배열의 기본 연산
# Max: 4
# Average: 2.0
# Max index: 2

print("[3,3] 랜덤 숫자로 이루어진 행렬 만들기")
c = np.random.rand(3,3)
print(c)

# [[0.3685617  0.92379794 0.85724741]
#  [0.31633374 0.81297231 0.63349589]
#  [0.96987799 0.7984839  0.28508072]]

print("배열의 차원 출력하기")
print("Shape of a:{}".format(a.shape))
print("Shape of b:{}".format(b.shape))
print("Shape of c:{}.".format(c.shape))

#Shape of a:(3,)
#Shape of b:(5,)
#Shape of c:(3, 3).
```

**Lessen 02** : [CodeLab](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l02c01_celsius_to_fahrenheit.ipynb#scrollTo=_wJ2E7jV5tN5)

머신러닝을 활용한 미니 프로젝트를 진행한다.

#### The Basics Training

~~~python
# 섭씨를 화씨로 바꾸기 (Celsius to Fahrenheit)
# 일반적인 공식은 다음과 같다
# f = c × 1.8 + 32
# 그러나 공식을 이용하지 않고 머신러닝을 통해서 해결하기

import tensorflow as tf
import numpy as np
#1 트레이닝 데이터 설정

celsius_q = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i,c in enumerate(celsius_q):
  print("{} 섭씨온도 = {} 화씨온도".format(c, fahrenheit_a[i]))

#2 모델 만들기
# 간단하고 사용하기 쉬운 모델을 만든다.
# 문제가 간단해서 단일 뉴런이 있는 단일 레이어를만 필요하다 (Dense 네트워크를 사용)

# input_shape=[1] : 레이어에 대한 입력을 단일 값으로 지정한다.
# 현재 레이어의 입력은 값이 한개인 1차원 배열의 형태이다. (Celcius를 나타내는 부동 소수점 숫자)

# units=1 : 레이어의 뉴런 수를 지정한다.
# 현재 레이어가 최종 레이어라서 화씨 온도를 나타내는 단일 부동 소수점 숫자를 나타낸다.
IO = tf.keras.layers.Dense(units=1, input_shape=[1])

#3 모델에 레이어 조립하기
model = tf.keras.Sequential([IO])
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(units=1, input_shape=[1])
# ])

#4 손실과 최적 함수를 이용해 모델 컴파일 하기
# 손실 함수 (Loss function) - 예측 결과가 원하는 결과로부터 얼마나 떨어져 있는지 측정하는 방법.
# 최적화 함수 - 손실을 줄이기 위해 내부 값을 조정하는 방법입니다.

# 여기에서 사용된 'mean_squared_error' 평균 제곱 오차의 손실 함수이다.
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0,1))


#4 모델 훈련
# fit 메서드를 통해 모델을 교육한다.
# epochs 인수는 사이클 실행 횟수를 의미, verbose는 메서드가 생성하는 출력 양을 조절한다.
history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("모델 훈련 완료")
~~~
