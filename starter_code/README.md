# Run main script (Instructions)

## For training 

```python
  python main.py --mode "train" --data_dir "./data" --save_dir "./" 
```

## For testing

```python
  python main.py --mode "test" --data_dir "./data" --save_dir "./" 
```

## For predictions

```python
  python main.py --mode "predict" --data_dir "./data" --save_dir "./" 
```


[dependencies]
python = "^3.7"
tensorflow = "2.2.0"
tensorflow-datasets = "^4.1.0"
matplotlib = "^3.3.3"
autokeras = "^1.0.11"
keras-tuner = {git = "https://github.com/keras-team/keras-tuner.git", rev = "1.0.2rc3"}