def mySGN(t, h):
    s = sum(t[i] * h[i] for i in range(len(t)))
    if s >= 0: return 1
    else: return -1

def main():
    w = [1, 2, -10]
    x = [3, 4, -1]

    y = mySGN(w, x)

    print("wTx =", sum(w[i] * x[i] for i in range(len(w))))
    print("Prediction:", y)

if __name__ == "__main__":
    main()