#* using pre-trained models to construct graphs

cd KGE
## gowalla
python construct_loc_loc_graph.py --model_type transe --dataset gowalla --pretrain_model ../data/pretrained-model/gowalla_scheme2/gowalla-transe-1695871810.ckpt --version scheme2 --threshold 100 --user_count 10919 --loc_count 35160
## foursquare-SIN
python construct_loc_loc_graph.py --model_type transe --dataset foursquare --pretrain_model ../data/pretrained-model/foursquare_scheme2/foursquare-transe-1694521793.ckpt --version scheme2 --threshold 20 --user_count 4638 --loc_count 9731

## NYC
python construct_loc_loc_graph.py --model_type transe --dataset NYC --pretrain_model ../data/pretrained-model/nyc_scheme2/nyc-transe-1701600057.ckpt --version scheme2 --threshold 10 --user_count 1083 --loc_count 9989

## TKY
python construct_loc_loc_graph.py --model_type transe --dataset TKY --pretrain_model ../data/pretrained-model/tky_scheme2/tky-transe-1701600148.ckpt --version scheme2 --threshold 10 --user_count 2293 --loc_count 15177



# #* user-POI graph for reference but not used yet
# ## gowalla
# python construct_user_loc_graph.py --model_type transe --dataset gowalla --pretrain_model ../data/pretrained-model/gowalla_scheme2/gowalla-transe-1695871810.ckpt --version scheme2 --threshold 100 --user_count 10919 --loc_count 35160
# ## foursquare
# python construct_user_loc_graph.py --model_type transe --dataset foursquare --pretrain_model ../data/pretrained-model/foursquare_scheme2/foursquare-transe-1694521793.ckpt --version scheme2 --threshold 20 --user_count 4638 --loc_count 9731