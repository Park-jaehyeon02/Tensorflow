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
        activation_fn (callable): Ȱ��ȭ �Լ�
    """

    def __init__(self, num_inputs, activation_fn):
        super().__init__()
        #���������� ����ġ ���Ϳ� ���Ⱚ�� �ʱ�ȭ��
        self.W = np.random(num_inputs)
        self.b = np.random.rand(1)
        self.activation_fn = activation_fn

    def forward(self,x):
        #������ ���� �Է� ��ȣ�� ����
        z = np.dot(x,self.W) + self.b
        return self.activation_fn(z)
