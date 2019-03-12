---
title: "Tensorflow Free Course"
categories: ML
tags: ML
---

Udacity의 [텐서플로우 무료 강좌](https://classroom.udacity.com/courses/ud187)가 있어서 공부하면서 정리를 해보려고 한다.

CodeLab에서 직접 실습하면서 배우는 방식으로 강의가 진행된다.

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

**Lesson 03** : [CodeLab](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l03c01_classifying_images_of_clothing.ipynb#scrollTo=vasWnqRgy1H4)

운동화와 셔츠와 같은 의류 이미지를 분류하는 신경망을 구축하교 교육하는 예제로 Fashion MNIST 데이터를 사용한다.

MNIST는 컴퓨터 비전을위한 기계학습 프로그램에서 "Hello World"와 같은 고전적인 데이터셋을 의미한다.

이러한 데이터를 통해 알고리즘이 제대로 동작하는지 코드를 테스트하고 디버깅하기 위한 용도로 사용한다.

<table>
  <tr><td>
    <img src="https://tensorflow.org/images/fashion-mnist-sprite.png"
         alt="Fashion MNIST sprite"  width="600">
  </td></tr>
  <tr><td align="center">
    <b>Figure 1.</b> <a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST samples</a> (by Zalando, MIT License).<br/>&nbsp;
  </td></tr>
</table>

#### Classifying Image of Clothing
~~~python
# 운동화와 셔츠와 같은 의류 이미지를 분류하는 신경망을 구축하교 교육하는 예제
!pip install -U tensorflow_datasets

import tensorflow as tf
import tensorflow_datasets as tfds
tf.logging.set_verbosity(tf.logging.ERROR)

# 파이썬 유틸라이브러리
import math
import numpy as np
import matplotlib.pyplot as plt

# 현재 작업의 진행 상황을 알려주는 라이브러리
import tqdm
import tqdm.auto
tqdm.tqdm = tqdm.auto.tqdm

# 70000개의 이미지 중 60000개는 train 데이터로 사용하고
# 나머지 10000개는 평가용 이미지로 사용한다.
# 모델이 학습할 때 train_dataset을 사용한다.
# 모델이 학습할 때 test_dataset을 사용한다.
# 각 이미지는 [0, 255]의 범위를 갖는 28 x 28 배열이다.
dataset, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

~~~

![img](data:image/png;base64,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)

~~~python
# 각 이미지들은 다음 라벨과 매핑된다.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal',      'Shirt',   'Sneaker',  'Bag',   'Ankle boot']

# 모델 학습 이전에 데이터셋의 포맷을 확인한다.
num_train_examples = metadata.splits['train'].num_examples
num_test_examples = metadata.splits['test'].num_examples
print("Number of training examples: {}".format(num_train_examples))
print("Number of test examples:     {}".format(num_test_examples))
# Number of training examples: 60000
# Number of test examples:     10000

# 데이터 전처리
# 이미지 데이터의 각 픽셀의 값은 [0~255] 범위릐 값이다.
# 실제 모델이 동작하기 위해서 [0,1] 범위로 표준화 해야한다.
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

# map 함수는 데이터의 각 요소에 normalize 함수를 적용한다.
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)
~~~