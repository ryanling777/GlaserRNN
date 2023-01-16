import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('Test_Output.csv')

#print(df.head(5))
#print(df.hand_model_output.values[:100])

fig, ax = plt.subplots()

#print(df.hand_model_output.values[:100])

for arr in df.hand_model_output.values[:100]:
    print(*arr.T)
   
    #ax.scatter(*arr.T, alpha = 0.1, color = 'tab:blue')
    

ax.set_title('model output')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.savefig('Model Output.png')