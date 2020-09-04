## Benchmarking models with MOSES
Models can be easily benchmarked using MOSES. To do this, we recommend reading the MOSES documentation, available at https://github.com/molecularsets/moses. If you want to compare to previously benchmarked models you will have to train models using the MOSES datasets, available [here](https://github.com/molecularsets/moses/tree/master/data).

Once you have a satisfactorily trained model, you can run a Generation job to create 30,000 new structures (see [2_using_a_new_dataset](./2_using_a_new_dataset.md) and follow the instructions using the MOSES dataset). The generated structures can then be used as the <generated dataset> in MOSES evaluation jobs.
