import matplotlib.pyplot as plt
import numpy as np

# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
markes = ['-p', '-s', '-^', 'go-']
dimension = [20, 40, 60, 80, 100]

our = [0.1805, 0.2665, 0.3392, 0.1612, 0.1998, 0.2233]
#* graph regularization
foursquare_time_kl_reg_0001 = [0.1751, 0.2572, 0.3327, 0.1549, 0.1917, 0.2160]    
foursquare_time_kl_reg_001 = [0.1751, 0.2579, 0.3344, 0.1549, 0.1917, 0.2162]
foursquare_time_kl_reg_01 = [0.1723, 0.2551, 0.3260, 0.1522, 0.1890, 0.2119]
foursquare_time_kl_reg_10 = [0.1779, 0.2635, 0.3435, 0.1580, 0.1962, 0.2220]
foursquare_time_kl_reg_acc1 = [0.1751, 0.1751, 0.1723, 0.1805, 0.1779] #* 选recall@1与recall@5
foursquare_time_kl_reg_acc5 = [0.2572, 0.2579, 0.2551, 0.2665, 0.2635]
# plt.plot(dimension, foursquare_time_kl_reg_acc1, markes[3], label='Acc@1', ms=10)
# plt.plot(dimension, foursquare_time_kl_reg_acc5, markes[1], label='Acc@5', ms=10)
# plt.xlabel("Weight of temporal regularization", fontsize=20)
# plt.ylabel("Acc", fontsize=20)
# plt.axis(ymin=0.17, ymax=0.27)
# plt.yticks(np.arange(0.17, 0.2701, 0.02), fontsize=20)
# plt.xticks(dimension, [0.001, 0.01, 0.1, 1, 10], fontsize=20)
# plt.grid(linestyle='--', linewidth=1, alpha=0.3) # color='r', 
# plt.tight_layout()
# plt.legend(fontsize=20, frameon=False)
# plt.savefig('Temporal_Regularization_Acc@1-5.pdf')
foursquare_time_kl_reg_ndcg1 = [0.1549, 0.1549, 0.1522, 0.1612, 0.1580] #* 选ndcg@1与ndcg@5
foursquare_time_kl_reg_ndcg5 = [0.1917, 0.1917, 0.1890, 0.1998, 0.1962]
plt.plot(dimension, foursquare_time_kl_reg_ndcg1, markes[3], label='NDCG@1', ms=10)
plt.plot(dimension, foursquare_time_kl_reg_ndcg5, markes[1], label='NDCG@5', ms=10)
plt.xlabel("Weight of temporal regularization", fontsize=20)
plt.ylabel("NDCG", fontsize=20)
plt.axis(ymin=0.15, ymax=0.21)
plt.yticks(np.arange(0.15, 0.2101, 0.01), fontsize=20)
plt.xticks(dimension, [0.001, 0.01, 0.1, 1, 10], fontsize=20)
plt.grid(linestyle='--', linewidth=1, alpha=0.3) # color='r', 
plt.tight_layout()
plt.legend(fontsize=20, frameon=False)
plt.savefig('Temporal_Regularization_NDCG@1-5.pdf')

foursquare_dis_kl_reg_0001 = [0.1809, 0.2633, 0.3312, 0.1624, 0.1996, 0.2216]
foursquare_dis_kl_reg_001 = [0.1815, 0.2661, 0.3368, 0.1620, 0.2001, 0.2227]
foursquare_dis_kl_reg_01 = [0.1787, 0.2633,0.3323, 0.1604, 0.1989, 0.2211]
foursquare_dis_kl_reg_10 = [0.1755, 0.2656, 0.3366, 0.1558, 0.1962, 0.2191]

foursquare_contra_kl_reg_0001 = [0.1742, 0.2643, 0.3396, 0.1562, 0.1966, 0.2209]
foursquare_contra_kl_reg_001 = [0.1729, 0.2628, 0.3381, 0.1539, 0.1942, 0.2184]
foursquare_contra_kl_reg_01 = [0.1738, 0.2510, 0.3238, 0.1529, 0.1875, 0.2111]
foursquare_contra_kl_reg_10 = [0.1762, 0.2581, 0.3325, 0.1549, 0.1920, 0.2159]

foursquare_anchor_100 = [0.1790, 0.2615, 0.3348, 0.1577, 0.1946, 0.2183]
foursquare_anchor_200 = [0.1725, 0.2594, 0.3331, 0.1528, 0.1917, 0.2153]
foursquare_anchor_1000 = [0.1764, 0.2600, 0.3389, 0.1561, 0.1936, 0.2190]
foursquare_anchor_2000 = [0.1753, 0.2598, 0.3353, 0.1550, 0.1928, 0.2172]

foursquare_gcn_1 = [0.1688, 0.2637, 0.3376, 0.1481, 0.1908, 0.2146]
foursquare_gcn_2 = [0.1785, 0.2667, 0.3385, 0.1571, 0.1968, 0.2200]
foursquare_gcn_4 = [0.1738, 0.2557, 0.3344, 0.1556, 0.1926, 0.2179]
foursquare_gcn_5 = [0.1725, 0.2544, 0.3284, 0.1523, 0.1887, 0.2125]

foursquare_tra_delta_01 = [0.1813, 0.2708, 0.3420, 0.1618, 0.2018, 0.2248]  #* tra选0或者0.1差不多, 默认是0
foursquare_tra_delta_02 = [0.1742, 0.2669, 0.3394, 0.1553, 0.1968, 0.2201]
foursquare_tra_delta_03 = [0.1792, 0.2637, 0.3376, 0.1584, 0.1964, 0.2203]
foursquare_tra_delta_04 = [0.1774, 0.2687, 0.3396, 0.1567, 0.1979, 0.2206]

foursquare_tem_delta_00 = [0.1785, 0.2622, 0.3333, 0.1590, 0.1965, 0.2196]
foursquare_tem_delta_01 = [0.1785, 0.2613, 0.3344, 0.1575, 0.1947, 0.2183]
foursquare_tem_delta_02 = [0.1781, 0.2637, 0.3413, 0.1589, 0.1973, 0.2224]
foursquare_tem_delta_03 = [0.1781, 0.2650, 0.3379, 0.1586, 0.1979, 0.2216]  #* tem选0.4, 默认是0.4

foursquare_dis_delta_01 = [0.1833, 0.2693, 0.3411, 0.1639, 0.2023, 0.2255]  #* dis选0.1 可能是随机数的原因导致的结果偏高, 就用默认的0吧, 画图不选recall@2
foursquare_dis_delta_02 = [0.1787, 0.2697, 0.3385, 0.1592, 0.2002, 0.2225]
foursquare_dis_delta_03 = [0.1856, 0.2689, 0.3370, 0.1633, 0.2007, 0.2227]
foursquare_dis_delta_04 = [0.1824, 0.2704, 0.3411, 0.1624, 0.2022, 0.2248]

# gowalla_acc5 = [0.3338, 0.3376, 0.3385, 0.3403, 0.3425]  # 10, 20, 30, 50, 100 Transition
# gowalla_acc10 = [0.4246, 0.4249, 0.4237, 0.4247, 0.4256]  # Preference

# foursquare_acc5 = [0.5727, 0.5757, 0.5723, 0.5748, 0.5747]  # 20, 50, 100  Transition
# foursquare_acc10 = [0.6512, 0.6513, 0.6512, 0.6512, 0.6514]  # Preference 缺30

# # Figure 6 (b)
# # plt.plot(dimension, gowalla_acc10, markes[1], label='Gowalla', ms=15)
# # plt.plot(dimension, foursquare_acc10, markes[2], label='Foursquare', ms=15)
# # plt.xlabel("Neighbors", fontsize=20, fontweight='bold')
# # plt.ylabel("Acc@10", fontsize=20, fontweight='bold')

# # Figure 6 (a)
# plt.plot(dimension, gowalla_acc5, markes[3], label='Gowalla', ms=15)
# plt.plot(dimension, foursquare_acc5, markes[1], label='Foursquare', ms=15)
# plt.xlabel("Neighbors", fontsize=20, fontweight='bold')
# # plt.ylabel("Acc@10", fontsize=20, fontweight='bold')
# plt.ylabel("Acc@5", fontsize=20, fontweight='bold')

# # Figure 6 (a)
# plt.axis(ymin=0.3, ymax=0.6)
# plt.yticks(np.arange(0.3, 0.6001, 0.05), fontsize=20, fontweight='bold')

# # Figure 6 (b)
# # plt.axis(ymin=0.4, ymax=0.7)
# # plt.yticks(np.arange(0.4, 0.7001, 0.05), fontsize=20, fontweight='bold')

# plt.xticks(dimension, [10, 20, 30, 50, 100], fontsize=20, fontweight='bold')

# # plt.grid(True)
# plt.grid(linestyle='--', linewidth=1, alpha=0.3) # color='r', 
# plt.tight_layout()
# plt.legend(fontsize=20, frameon=False)
# # plt.savefig('revise_analysis_Preference_Acc10.pdf')  # 之前是new_analysis_Preference_Acc10.pdf
# # plt.savefig('revise_analysis_POI_Acc5.pdf')
# plt.savefig('test.pdf')
# plt.show()
