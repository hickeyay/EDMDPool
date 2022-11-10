### Requirements
* networkx==2.5.1
* python==3.6.13
* pytorch==1.4.0
* torch-scatter==2.0.3
* torch-sparse==0.5.1
* torch-cluster==1.5.2
* torch-spline-conv==1.2.0
* torch-geometric==1.7.0
### Run
    ./run_EDMDPool.sh DATA FOLD GPU
to run on dataset using fold number (1-10).
You can run
```python
    ./run_EDMDPool.sh DD 0 0
```
to run on DD dataset with 10-fold cross validation on GPU #0.
    ./run_EDMDPool.sh COLLAB 1 1
to run on COLLAB dataset with first fold on GPU #1.
