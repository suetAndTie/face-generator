import matplotlib.pyplot as plt

x = [0,1,5,10,15,20,25]
y = [0.01, 1.8971047731377857, 1.973677877217517, 2.1907745339791265, 2.4108357541054164, 2.15292942928771, 2.3189814378112446]
y2 = [3.3251909783960976,3.3251909783960976,3.3251909783960976,3.3251909783960976,3.3251909783960976,3.3251909783960976,3.3251909783960976]

plt.plot(x, y, '--o', color='black', label='Generated')
plt.plot(x, y2, '-', color='red', label='Real')

plt.title('Inception Score of Images')
plt.xlabel('Epoch')
plt.ylabel('Inception Score')
plt.legend(loc='lower right')

plt.xticks([0, 5, 10, 15, 20, 25])
plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3])

plt.show()
