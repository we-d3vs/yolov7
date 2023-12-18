Download vast cli tool
```
wget https://raw.githubusercontent.com/vast-ai/vast-python/master/vast.py -O vast; chmod +x vast;
```

Login the vast machine
```
ssh XXX  # copy the ssh command from vast instance page.
```

Copy training files from local machine to the host
```bash
./vast copy $PATH/filesa*.zip ${VAST_MACHINE_ID}:/data/
```

unzip the files in vast machine in SWAP dir
```bash
cd /data/
cd /dev/shm/
sudo mkdir private-dataset
cd private-dataset
unzip /data/*zip
```


Set up the training
```bash
cd ${HOME}
git clone https://github.com/WongKinYiu/yolov7
cd yolov7
python3 -m pip install -r requirements.txt
python3 -m pip install "pycocotools>=2.0"
# prepare data
mkdir we_data
# copy train and test coco cache files from local to vas
cd $YOLOV7_PATH
./vast copy $TRAIN.cache ${VAST_MACHINE_ID}:~/yolov7/we_data/
./vast copy $TEST.cache ${VAST_MACHINE_ID}:~/yolov7/we_data/
# copy configs
./vast copy data/we_v104.yaml ${VAST_MACHINE_ID}:~/yolov7/data/
./vast copy cfg/training/yolov7_we.yaml ${VAST_MACHINE_ID}:~/yolov7/cfg/training/
```

Modify the code
```
cd $YOLOV7_PATH
./vast copy utils/datasets.py ${VAST_MACHINE_ID}:~/yolov7/utils/
./vast copy utils/loss.py ${VAST_MACHINE_ID}:~/yolov7/utils/
```

Change `utils/datasets.py` image base path: replace this
```
img = imread_fallback("/media/nas/private-dataset/" + path)
```
by
```
img = imread_fallback("/dev/shm/private-dataset/" + path)
```

Train the model
```bash
nice -n1 python3 -m torch.distributed.launch --nproc_per_node $N_GPUS train.py --workers $WORKERS \
--device 0,1,2,3 --sync-bn --batch-size 256 --data data/we_v104.yaml \
--img 640 640 --cfg cfg/training/yolov7_we.yaml --name yolov7-we-v104 \
--hyp data/hyp.scratch.p5.yaml \
--exist-ok --weights ""
```

