# eblict-nid-code-repo
code repo for eblict nid 

# Installation

## **For CPU-Inference**

**Environment Setup**

* **Installing conda**: Follow steps [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to install conda as needed
* **create a conda environment**: ```conda create -n nidocr python=3.8.13```
* **activate conda environment**: ```conda activate nidocr```


**CPU INFERENCE SERVER CONFIG**  

```python
OS          : Ubuntu 20.04.3 LTS       
Memory      : 23.4 GiB 
Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
Gnome       : 3.36.8
```

**STACK EXECUTION**

| **OP** | **avg-rough-Exec** | **Support** |
|  :----: |  :----:  |  :----:  |
| ocr initialization/loading | 6.5s-8.5s |5 iters |
| ocr execution | 4.5s-6.5s |5 iters |
