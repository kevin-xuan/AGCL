python main.py --gpu 1 --dataset four-sin --num_blocks 3 --num_heads 2 --dropout_rate 0.3 --l2_emb 1e-4 --layer_num 4
python main.py --gpu 1 --dataset gowalla  --num_blocks 3 --num_heads 2 --dropout_rate 0.2 --l2_emb 1e-3 --layer_num 2 # 不确定?

cd anchor-version
python main_anchor.py --gpu 2 --dataset four-sin --num_blocks 2 --num_heads 1 --dropout_rate 0.2 --l2_emb 1e-3 --layer_num 3  
python main_anchor.py --gpu 2 --dataset gowalla --num_blocks 2 --num_heads 1 --dropout_rate 0.2 --l2_emb 1e-3 --layer_num 2  