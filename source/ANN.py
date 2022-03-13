#-*- coding:cp949-*-
import numpy as np

class Neuron(object):
    """간단한 전방 전달 인공 뉴런
    Args:
        num_input (int): 입력 벡터 크기 / 입력 값 개수.
        activation_fn (callable): 활성화 함수
    Attributes:
        W (ndarray): 각 입력에 대한 가중치
        b (float): 편항값, 가중합에 더해짐
        size (int): 계층의 크기 / 뉴런의 개수
        activation_fn (callable): 활성화 함수
    """

    def __init__(self, num_inputs, layer_size ,activation_fn):
        super().__init__()
        #랜덤값으로 가중치 벡터와 편향값을 초기화함
        self.W = np.random.standard_normal((num_inputs),layer_size)
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn

    def forward(self,x):
        #뉴런을 통해 입력 신호를 전달
        z = np.dot(x,self.W) + self.b
        return self.activation_fn(z)
