import numpy as np
from mlp import MLP


def test_can_learn_xor():
    X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float32)
    y = np.array([0,1,1,0])
    model = MLP(2, 16, 2, seed=1)
    for _ in range(3000):
        logits = model.forward(X)
        grads = model.backward(y)
        model.step(grads, lr=0.1)
    pred = model.forward(X).argmax(1)
    assert (pred == y).all()


if __name__ == "__main__":
    test_can_learn_xor(); print("MLP OK")
