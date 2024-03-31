import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

def ars():
    plt.xlabel('Methods', fontsize=20, fontweight='bold')
    # plt.ylabel('ARI', fontsize=20, fontweight='bold')
    plt.ylabel('Acc@1', fontsize=20, fontweight='bold')

    # data = [0.13, 0.233, 0.305, 0.285, 0.385, 0.580, 0.510, 0.544, 0.478, 0.610]
    # data = [0.13, 0.233, 0.305, 0.285, 0.385, 0.580, 0.510, 0.544, 0.478, 0.638]
    # data = [0.13, 0.233, 0.305, 0.285, 0.385, 0.580, 0.510, 0.544, 0.478, 0.628]

    # Figure 4
    # data_acc = [0.1500, 0.1497, 0.1495, 0.1512, 0.1505, 0.1502]  # Figure 4(a)
    # data_mrr = [0.2391, 0.2394, 0.2376, 0.2422, 0.2412, 0.2394]  # Figure 4(b)
    # Figure 5
    data_acc = [0.2496, 0.2773, 0.2805, 0.2802, 0.2805]  # Figure 5(a)
    data_mrr = [0.3805, 0.4113, 0.4136, 0.4131, 0.4129]  # Figure 5(b)

    # labels = ['MV-PN', 'GAE', 'node2vec', 'HDGE', 'ZE-Mob', 'MV2vec', 'No-DSA', 'No-SemA', 'No-Geo', 'ARE']
    labels = ['NG', 'CG', 'EG', 'HG', 'RG']  # Figure 5
    # labels = ['Es1', 'Hs1', 'Rs1', 'Es2', 'Hs2', 'Rs2']  # Figure 4

    # plt.axhline(y=0.6, color='black', linestyle='--', linewidth=0.7)
    # Figure 4(a)
    # plt.axis(ymin=0.1494, ymax=0.1514)
    # plt.yticks(np.arange(0.1494, 0.1515, 0.0004), fontsize=20, fontweight='bold')
    # Figure 4(b)
    # plt.axis(ymin=0.2370, ymax=0.2430)
    # plt.yticks(np.arange(0.2370, 0.2430, 0.0012), fontsize=20, fontweight='bold')
    # plt.xticks(fontsize=18, rotation=90, fontweight='bold')

    # Figure 5(a)
    plt.axis(ymin=0.2450, ymax=0.2850)
    plt.yticks(np.arange(0.2450, 0.2851, 0.008), fontsize=20, fontweight='bold')
    
    # Figure 5(b)
    # plt.axis(ymin=0.37, ymax=0.42)
    # plt.yticks(np.arange(0.37, 0.4201, 0.01), fontsize=20, fontweight='bold')
    plt.xticks(fontsize=20, fontweight='bold')

    # plt.bar(range(len(data_acc)), data_acc, tick_label=labels, width=0.8,
    #         color=['mediumorchid', 'slategray', 'skyblue', 'lightseagreen', 'yellowgreen', 'olive', 'tan',
    #                'lightsalmon', 'tomato', 'red'])
    # plt.bar(range(len(data_acc)), data_acc, tick_label=labels, width=0.8,
    #         color=['olive', 'slategray', 'skyblue', 'tomato', 'yellowgreen'])
    # Figure 4
    # plt.bar(range(len(data_acc)), data_acc, tick_label=labels, width=0.8,
    #         color=['mediumorchid', 'slategray', 'skyblue', 'lightseagreen', 'yellowgreen', 'olive'])
    # Figure 5
    plt.bar(range(len(data_acc)), data_acc, tick_label=labels, width=0.8,
            color=['olive', 'slategray', 'skyblue', 'tomato', 'yellowgreen'])
    plt.tight_layout()
    # plt.savefig('revise_gowalla_contrast_transX_Acc.pdf')  # 之前是new_gowalla_contrast_transX_MRR.pdf Figure 4
    # plt.savefig('revise_foursquare_contrast_graph_Acc.pdf')  # Figure 5
    plt.show()


if __name__ == '__main__':
    ars()
