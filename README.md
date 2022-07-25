# eblict-nid-code-repo
code repo for eblict nid 

# Installation

## **For CPU-Inference**

**Environment Setup**

* **Installing conda**: Follow steps [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to install conda as needed
* **create a conda environment**: ```conda create -n nidocr python=3.8.13```
* **activate conda environment**: ```conda activate nidocr```
* **install cpu dependencies**  : ```./install_cpu.sh```  


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

## **For GPU-Inference**

**Environment Setup**

* **Installing conda**: 
    *  ```curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh```
    *  ```bash Miniconda3-latest-Linux-x86_64.sh```

* **create a conda environment**: ```conda create -n nidocr python=3.8.13```
* **activate conda environment**: ```conda activate nidocr```
* **install gpu dependencies**  : ```./install_gpu.sh```  

```
cd weights/
gdown 1gbCGRwZ6H0TO-ddd4IBPFqCmnEaWH-z7
gdown 1YwpcDJmeO5mXlPDj1K0hkUobpwGaq3YA
cd ..
python setup_check.py
```


**CPU INFERENCE SERVER CONFIG**  

```python
OS          : Ubuntu 20.04.4 LTS       
Memory      : 31.3 GiB 
Processor   : Intel® Core™ i9-10900K CPU @ 3.70GHz × 20    
Graphics    : NVIDIA GeForce RTX 3090/PCIe/SSE2
Gnome       : 3.36.8
```

<!-- **STACK EXECUTION**

| **OP** | **avg-rough-Exec** | **Support** |
|  :----: |  :----:  |  :----:  |
| ocr initialization/loading | 6.5s-8.5s |5 iters |
| ocr execution | 4.5s-6.5s |5 iters | -->
