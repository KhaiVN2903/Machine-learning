import numpy as np

def mySGN(t, h):
    s = np.dot(t, h)
    if s >= 0: return 1
    else: return -1

def neW(t, h, r):
    w_new = t + h*r
    return w_new

def main():
    w = np.array([-2, 1, 0])
    x = np.array([2, 3, 1])

    y = 1
    y_predict = mySGN(w, x)

    print("wTx =", np.dot(w, x))
    print("Prediction = ", y)

    if(y_predict != y):
        w = neW(w, y, x)
        print("Prediction is wrong. Updating w ...")
        print("New w = ", w)
        print("New wTx =", np.dot(w, x))

if __name__ == "__main__":
    main()