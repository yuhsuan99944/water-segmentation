# Water Segmentation
Water segmentation provides crucial information for flood prevention, water resource management, and urban planning. However, in remote sensing satellite imagery, accurately identifying Water-Land Interface Zone and edge information between water and land poses a challenge for traditional water segmentation methods. The objective of this article is to achieve more accurate segmentation results and identify interface zones between water and land. Simultaneously, reducing computational complexity to achieve real-time performance and suitability for edge devices. This paper combines CNN and Transformer to simultaneously capture local and global features. This paper proposes multi-scale projection and spatial reduction and attention in the multi-head attention mechanism, enabling the learning of different receptive fields and rich features. Using multi-spectral and color attributes to enhance the model's recognition capabilities. Furthermore, this paper proposed loss function for learning and supervision of edge information between different classes in the images. 

## Research Target
![image](https://github.com/yuhsuan99944/water-segmentation/assets/95406938/a6ff0b2e-f669-4dea-8b9d-7df581d00aa7)
![image](https://github.com/yuhsuan99944/water-segmentation/assets/95406938/d41934dc-cdea-4af2-aa1a-682a930deff0)



## Experiment Result
The images shown in Figure 1 represent the validation results of segmentation for small and large water areas. The dark areas indicate the ground truth, while the light-colored transparent regions represent the segmentation results of other state-of-the-art models. The popular models in traditional CNN and emerging technology CNN add Transformer are selected for comparison.

![image](https://github.com/yuhsuan99944/water-segmentation/assets/95406938/915c87f8-8bf8-4e79-b002-7f2a3273fbc7)

Figure1 The segmentation results for the small and large water area
