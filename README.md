### Matrix factorization techniques for recommender systems
---

### Dataset
I took dataset from [here](https://github.com/caserec/Datasets-for-Recommneder-Systems/tree/master/Processed%20Datasets/BookCrossing).  
Please, refer to above link if you want description.  

###### book_ratings.dat
|user|item |rating|
|:--:|:---:|:----:|
|1   |6264 |7     |
|1   |4350 |7     |
|1   |6252 |5     |
|1   |202  |9     |
|1   |6266 |6     |
|... |...  |...   |
|2945|15719|8     |
|2945|11960|6     |
|2945|8515 |9     |
|2945|9417 |7     |
|2945|8052 |8     |

---

### Experimental Environment
- numpy == 1.16.4
- pandas == 0.24.2
- tensorflow == 1.14.0-rc1
- Colab
---

### How to Use
**Execution on cpu**  
config.json
```json
{
  "mode": "cpu",
  "data_path": "data/book_ratings.dat",
  "epoch": 500,
  "alpha": 1e-2,
  "beta": 1e-2,
  "dim": 2,
  "num_rec_items": 3,
  "verbose": 1
}
```

```
python main.py
```

**Execution on gpu**  
config.json
```json
{
  "mode": "gpu",
  "data_path": "data/book_ratings.dat",
  "epoch": 500,
  "alpha": 1e-3,
  "beta": 1e-4,
  "dim": 2,
  "num_rec_items": 3,
  "verbose": 2
}
```

```
python main.py
```
---


### License
MIT

---


### Author
[rnaster](https://github.com/rnaster)
