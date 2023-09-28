#* using pre-trained models to construct graphs

cd KGE
# TransE
## gowalla
python construct_loc_loc_graph.py --model_type transe --dataset gowalla --pretrain_model ../data/pretrained-model/gowalla_scheme2/gowalla-transe-1695871810.ckpt --version scheme2 --threshold 100 --user_count 10919 --loc_count 35160
## foursquare
python construct_loc_loc_graph.py --model_type transe --dataset foursquare --pretrain_model ../data/pretrained-model/foursquare_scheme2/foursquare-transe-1694521793.ckpt --version scheme2 --threshold 20 --user_count 4638 --loc_count 9731

## gowalla
python construct_user_loc_graph.py --model_type transe --dataset gowalla --pretrain_model ../data/pretrained-model/gowalla_scheme2/gowalla-transe-1695871810.ckpt --version scheme2 --threshold 100 --user_count 10919 --loc_count 35160
## foursquare
python construct_user_loc_graph.py --model_type transe --dataset foursquare --pretrain_model ../data/pretrained-model/foursquare_scheme2/foursquare-transe-1694521793.ckpt --version scheme2 --threshold 20 --user_count 4638 --loc_count 9731