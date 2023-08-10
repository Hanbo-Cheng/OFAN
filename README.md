# Optical Flow Aware Network
Offical implement of Optical Flow Aware Network

unzip the `CROHME.zip`

```
unzip CROHME.zip
```


## train
training WAP-OFAN:
```
python train_optical.py
```


training WAP only with AOFM (online modality):
```
python train_optical_single.py
```

training original WAP (offline modality):
```
python train_wap.py
```

## test

testing WAP-OFAN:
```
bash test_on.sh
```

testing test_on_single.sh:
```
bash test_on_single.sh
```

testing original WAP:
```
bash test.sh
```


