# graph regularization
python main.py --time_kl_reg 0.001 --gpu 0
python main.py --time_kl_reg 0.01 --gpu 1
python main.py --time_kl_reg 0.1 --gpu 2
python main.py --time_kl_reg 10 --gpu 3

python main.py --dis_kl_reg 0.001 --gpu 0
python main.py --dis_kl_reg 0.01 --gpu 1
python main.py --dis_kl_reg 0.1 --gpu 2
python main.py --dis_kl_reg 10 --gpu 3

python main.py --contra_reg 0.001 --gpu 0
python main.py --contra_reg 0.01 --gpu 1
python main.py --contra_reg 0.1 --gpu 2
python main.py --contra_reg 10 --gpu 3

# anchor number
python main.py --anchor_num 100 --gpu 0
python main.py --anchor_num 200 --gpu 1
python main.py --anchor_num 1000 --gpu 2
python main.py --anchor_num 2000 --gpu 3

# GCN layer
python main.py --layer_num 1 --gpu 0
python main.py --layer_num 2 --gpu 1
python main.py --layer_num 4 --gpu 2
python main.py --layer_num 5 --gpu 3

# adaptive graph threshold
python main.py --tra_delta 0.1 --gpu 0
python main.py --tra_delta 0.2 --gpu 1
python main.py --tra_delta 0.3 --gpu 2
python main.py --tra_delta 0.4 --gpu 3

python main.py --tem_delta 0.0 --gpu 3
python main.py --tem_delta 0.1 --gpu 0
python main.py --tem_delta 0.2 --gpu 1
python main.py --tem_delta 0.3 --gpu 2

python main.py --dis_delta 0.1 --gpu 0
python main.py --dis_delta 0.2 --gpu 1
python main.py --dis_delta 0.3 --gpu 2
python main.py --dis_delta 0.4 --gpu 3

