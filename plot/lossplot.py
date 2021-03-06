import matplotlib.pyplot as plt

x = range(1, 26)

gloss = [0.030, 0.029, 0.026, 0.025, 0.024, 0.024, 0.024, 0.023, 0.023, 0.023,
            0.023, 0.023, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022, 0.022,
            0.022, 0.022, 0.022, 0.022, 0.022]

dloss = [0.075, 0.057, 0.049, 0.047, 0.045, 0.045, 0.044, 0.043, 0.043, 0.043,
             0.043, 0.043, 0.043, 0.043, 0.043, 0.042, 0.042, 0.043, 0.043, 0.042,
             0.042, 0.042, 0.042, 0.042, 0.042]
bconverge = [0.086, 0.064, 0.053, 0.051, 0.048, 0.048, 0.047, 0.046, 0.046, 0.046,
                0.046, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045, 0.045,
                 0.045, 0.045, 0.045, 0.045, 0.045]

plt.plot(x, gloss, label='G Loss')
plt.plot(x, dloss, label='D Loss')
plt.plot(x, bconverge, label='BEGAN Convergence')

plt.title('Metrics of BEGAN')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend(loc='upper right')

plt.show()
