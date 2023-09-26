# 先融入temporal transition prior
python main.py --time_prompt False --dis_prompt False --dis_kl_reg 0. --contra_reg 0. --gpu 0
# # 再对比一下先融入spatial proximity prior,应该与temporal prior一致
# python main.py --time_prompt False --dis_prompt False --time_kl_reg 0. --contra_reg 0. --gpu 1
# 然后加入spatial prompt
python main.py --time_prompt False --dis_kl_reg 0. --contra_reg 0. --gpu 2
# 然后融入spatial proximity prior
python main.py --time_prompt False  --contra_reg 0. --gpu 3
# 再然后加入time prompt
python main.py --contra_reg 0. --gpu 0
# 最后加入图对比loss
python main.py --gpu 1
# # 后面再跑一个使用same anchor_idx
# python main.py --same_anchor True --gpu 2