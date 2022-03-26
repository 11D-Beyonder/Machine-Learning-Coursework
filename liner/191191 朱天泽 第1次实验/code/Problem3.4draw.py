import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = ax1.twinx()
ax2.axis('off')

tip = pd.DataFrame({'size of subset': [2, 3, 5, 10, 15, 30, 50],
                    'error rate': [0.06999999999999995, 0.050000000000000044, 0.040000000000000036,
                                   0.040000000000000036, 0.040000000000000036, 0.040000000000000036,
                                   0.040000000000000036]})
sns.barplot(x='size of subset', y='error rate', data=tip, ax=ax1, alpha=0.63)
sns.pointplot(x='size of subset', y='error rate', data=tip, ax=ax2, color='mediumpurple')
plt.show()
