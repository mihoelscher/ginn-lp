# GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations

This code is an implementation of the AAAI 2024 paper "GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations".
[[ArXiv](https://arxiv.org/abs/2312.10913)]

GINN-LP is a an end-to-end differentiable interpretable neural network that can recover mathematical expressions that take the form of multivariate Laurent polynomial (LP) equations. This is made possible by a new type of interpretable neural network block, named the power-term approximator (PTA). 

<img src="./assets/GINN-LP PTA Block.png" width=480, alt="GINN-LP Architecture">

LPs produce equations important in physics and real-world systems, such as the Coulomb's law, the gravitational potential energy, and the kinetic energy. 

- $E_f = \frac{1}{4\pi\epsilon}\frac{q_1q_2}{r^2}$ 
- $K = \frac{m}{2}(u^2 + v^2 + w^2)$ 
- $U = Gm_1m_2(\frac{1}{r_2} - \frac{1}{r_1})$

GINN-LP can discover these equations from data, without any prior knowledge of the underlying equations.



<img src="./assets/GINN-LP Architecture.png" width=1080, alt="GINN-LP Architecture">

## Discovering an equation using GINN-LP

Below, we outline the steps to discover the Coulomb's law equation using GINN-LP. The data used for this example is available in the data folder.

1. Install dependencies
    ```bash
    pip install -r requirements.txt
    ```
   
2. Run the code
    ```bash
   python run.py --data data/feynman_I_12_2 --format tsv
    ```
Here, the data file should be a csv or tsv with any number of feature columns and a target column. The target column should be named as "target".

## Citation
If our work is useful, please consider citing:

```
  @article{ranasinghe2023ginn,
    title={GINN-LP: A Growing Interpretable Neural Network for Discovering Multivariate Laurent Polynomial Equations},
    author={Ranasinghe, Nisal and Senanayake, Damith and Seneviratne, Sachith and Premaratne, Malin and Halgamuge, Saman},
    journal={arXiv preprint arXiv:2312.10913},
    year={2023}
  }
```

