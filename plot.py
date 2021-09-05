import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pyplotz.pyplotz import plt

plt.style.use(['science', 'ieee'])

df = pd.read_csv("data/data.csv")

# fig = plt.figure(figsize=(16, 48))
# plt.suptitle('各点元素浓度', fontsize=40)
# for idx, col in tqdm(enumerate(('AU', 'B', 'SN', 'CU', 'AG', 'BA', 'MN', 'PB', 'ZN', 'AS',
#        'SB', 'BI', 'HG', 'MO', 'W', 'F'))):
#
#     color = df[col]
#     ax = fig.add_subplot(8, 2, idx+1)
#     plt.scatter(df['X'], df['Y'], c=color)
#     plt.xticks([])
#     plt.yticks([])
#     plt.axis('off')
#     ax.set_title(col, fontsize=40)

# plt.savefig("eda.jpg", dpi=200)
for idx, col in tqdm(enumerate(('AU', 'B', 'SN', 'CU', 'AG', 'BA', 'MN', 'PB', 'ZN', 'AS',
                                'SB', 'BI', 'HG', 'MO', 'W', 'F'))):
    color = df[col]
    plt.scatter(df['X'], df['Y'], c=color, s=4)
    plt.xticks([])
    plt.yticks([])
    #     plt.axis('off')
    # plt.title(f"{col}", fontsize=18)
    plt.savefig(f'/Users/blackstar/Desktop/graduation/mypaper/sysu-graduate-thesis-master/figures/{col}.jpg', dpi=300)
    # plt.show()