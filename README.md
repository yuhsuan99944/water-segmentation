# water-segmentation

Water segmentation provides crucial information for flood prevention, water resource management, and urban planning. However, in remote sensing satellite imagery, accurately identifying Water-Land Interface Zone and edge information between water and land poses a challenge for traditional water segmentation methods. The objective of this article is to achieve more accurate segmentation results and identify interface zones between water and land. Simultaneously, reducing computational complexity to achieve real-time performance and suitability for edge devices. This paper combines CNN and Transformer to simultaneously capture local and global features. This paper proposes multi-scale projection and spatial reduction and attention in the multi-head attention mechanism, enabling the learning of different receptive fields and rich features. Using multi-spectral and color attributes to enhance the model's recognition capabilities. Furthermore, this paper proposed loss function for learning and supervision of edge information between different classes in the images. In the inference of water segmentation results, the proposed method outperforms advanced model with an improvement of 10.1% in mIoU and 6.7% in mean F1 score. 