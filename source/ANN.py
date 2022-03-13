#-*- coding:cp949-*-
import numpy as np

class Neuron(object):
    """������ ���� ���� �ΰ� ����
    Args:
        num_input (int): �Է� ���� ũ�� / �Է� �� ����.
        activation_fn (callable): Ȱ��ȭ �Լ�
    Attributes:
        W (ndarray): �� �Է¿� ���� ����ġ
        b (float): ���װ�, �����տ� ������
        size (int): ������ ũ�� / ������ ����
        activation_fn (callable): Ȱ��ȭ �Լ�
    """

    def __init__(self, num_inputs, layer_size ,activation_fn):
        super().__init__()
        #���������� ����ġ ���Ϳ� ���Ⱚ�� �ʱ�ȭ��
        self.W = np.random.standard_normal((num_inputs),layer_size)
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_fn = activation_fn

    def forward(self,x):
        #������ ���� �Է� ��ȣ�� ����
        z = np.dot(x,self.W) + self.b
        return self.activation_fn(z)
