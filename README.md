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
- numpy
- pandas
- tensorflow
- matplotlib
---

### Description
I refer to [this paper](https://dl.acm.org/citation.cfm?id=1608614).  
I implemented this paper with numpy and keras.  
It is executed to numpy or keras according to CPU or GPU.  
Seed is fixed for achieving reproducible model.  
These two models have difference for the number of trainable parameter.  
For example, suppose latent dimension is 2.  
In the numpy model if the number of user and item is 100 and 1000, then the number of trainable parameter is 100 * 2 + 1000 * 2.   
But, in the keras model if user, item`s ID max is 250 and 1300 despite the number of user and item, then it is 250 * 2 + 1300 * 2.  
Scatter plot for latent vectors is drawn when latent dimension is only 2.  
After training, rating prediction is written to csv, and do recommend items for unobserved item.  

---

### How to Use
##### Simple usage
**Execution on cpu**
```
python main.py --data_dir data/book_ratings.dat
```

**Execution on gpu**
```
python main.py --data_dir data/book_ratings.dat --mode gpu
```

##### Other parameters
```
python main.py --data_dir data/book_ratings.dat --mode gpu --epoch 300 --l2_rate 1e-5 --latent_dim 2 --num_rec_items 3
```

---


### Experimental Result
##### Training and Evaluation
Numpy model takes adverse influence by data`s original scale.  
So, in this model it is converted to range 0 ~ 1.  
Keras model takes adverse influence by scale down.  
So, in this model original scale is used.  
Dataset is not split to training and test.  
All data is used to train the model.  
So, model evaluation is substituted to last cost.  
The cost is defined to [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation).  


##### Comparison and Result
###### numpy model
cost plot L2 rate 1e-1|L2 rate 1e+1|L2 rate 1e-1|L2 rate 0|
:--:|:--:|:--:|:--:|
![](asset/numpy/cost-plot-l2-1e-1.png)|![](asset/numpy/l2-1e+1.png)|![](asset/numpy/l2-1e-1.png)|![](asset/numpy/l2-1e-1.png)|

###### keras model
cost plot L2 rate 1e-5|L2 rate 1e-4|L2 rate 1e-5|L2 rate 1e-6|L2 rate 0|
:--------------------:|:----------:|:----------:|:----------:|:-------:|
![](asset/keras/cost-plot-l2-1e-5.png)|![](asset/keras/l2-1e-4.png)|![](asset/keras/l2-1e-5.png)|![](asset/keras/l2-1e-6.png)|![](asset/keras/l2-0.png)|

These two models are set to latent dimension = 2, learning_rate = 1e-3 and learn 2000 times.  
The numpy model\`s training time is 36 mins and the other is 7 mins.  
The more L2 regularization rate, latent vector\`s properties are shrinking.  
Since the numpy model\`s training speed is slow, the more training times is necessary.  
When [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) is used in the keras model the training speed is very slow.  
So, [Adam](https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Adam) is adopted.  
The following table shows **keras model**\`s training time and RMSE according to latent dimension.  
All experiment is set to epoch = 2000, L2 regularization = 1e-5.  

###### Results according to latent dimension
|dimension|training time(mm:ss)|RMSE  |
|:-------:|:------------------:|:----:|
|2        |07:41               |1.7128|
|10       |08:10               |0.8486|
|50       |09:35               |0.7808|
|100      |12:20               |0.7834|
|500      |20:31               |0.8093|


---


### License
MIT

---


### Author
[rnaster](https://github.com/rnaster)
