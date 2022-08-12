python3 ./train.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X4Gates_Circle_l100
python3 ./test_recurrent.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X4Gates_Circle_l100 -j8
python3 ./test_dataloader.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X4Gates_Circle_l100 -j8
python3 ./train_recurrent.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X4Gates_Circle_l100 -j2
python3 ./train.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X4Gates_Circle_l100 -f world -r run_world

python -m orb_imitation.datagen.NetworkTestClient -arc resnet8 -w /media/data2/teamICRA/runs/ResNet8_bs=32_lt=MSE_lr=0.001_c=baseline/best.pth
python -m orb_imitation.datagen.RaceNet8TestClient -arc racenet8 -w /media/data2/teamICRA/runs/RaceNet8-Saturday/best.pth
python -m orb_imitation.datagen.RaceNet8TestClient -arc racenet8 -w /media/data2/teamICRA/runs/RaceNet8_all=32_lt=MSE_lr=0.001_c=run0/best.pth
python3 -m orb_imitation.datagen.SimClient
python -m orb_imitation.datagen.RaceNet8TestClient -arc racenet8 -w /media/data2/teamICRA/runs/RaceNet8_all=32_lt=MSE_lr=0.001_c=run0/best.pth
python3 ./train_newloader.py -pb /media/data2/teamICRA -db /media/data2/teamICRA/X4Gates_Circles_rl18tracks -n X4Gates_Circle_l100
python3 -m orb_imitation.datagen.OnegateClient