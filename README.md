# Average gradient-based adversarial attack
This repository contains the code for Average gradient-based adversarial attack.

Deep neural networks (DNNs) are vulnerable to adversarial attacks which can fool the classifiers by adding small perturbations to the original example. The added perturbations in most existing attacks are mainly determined by the gradient of the loss function with respect to the current example. In this paper, a new average gradient-based adversarial attack is proposed. In our proposed method, via utilizing the gradient of each iteration in the past, a dynamic set of adversarial examples is constructed first in each iteration. Then, according to the gradient of the loss function with respect to all the examples in the constructed dynamic set and the current adversarial example, the average gradient can be calculated, which is used to determine the added perturbations. Different from the existing adversarial attacks, the proposed average gradient-based attack optimizes the added perturbations through a dynamic set of adversarial examples, where the size of the dynamic set increases with the number of iterations. Our proposed method possesses good extensibility and can be integrated into most existing gradient-based attacks. Extensive experiments demonstrate that, compared with the state-of-the-art gradient-based adversarial attacks, the proposed attack can achieve higher attack success rates and exhibit better transferability, which is helpful to evaluate the robustness of the network and the effectiveness of the defense method.

Requirements

- Python 3.6.5
- Tensorflow 1.12.0 
- Numpy 1.15.4 
- Opencv2 3.4.2

Running the code

 python mi_fagsm.py:  generate adversarial examples for Inception_V4 using MI-FAGSM.
 

Models

Inception_V3 
http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz

Inception_V4 
http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz

Inception_ResNet_V2 
http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

ResNet_V2_152 
http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz




Example usage

- Generate adversarial examples:
python mi_fagsm.py

