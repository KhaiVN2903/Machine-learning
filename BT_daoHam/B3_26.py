def grad(x):
    return 2*x - 4

def cost(x):
    return x*x - 4*x + 5

def myGD1(x0):
    x = [x0]

    for i in range(4):
        x_new = x[-1] - 0.2*grad(x[-1])
        x.append(x_new)
    
    return x

def main():
    x = myGD1(5)
    for i in range(5):
        print('Solution x = %f, cost = %f, after %d iterations' % (x[i], cost(x[i]), i))


if __name__ == "__main__":
    main()