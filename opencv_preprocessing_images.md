```python
from src.visualizations import *

compare_original_and_processed('data')
```


    
![png](test_files/test_0_0.png)
    



```python
plot_image_with_bboxes('data')
```


    
![png](test_files/test_1_0.png)
    



```python
from src.utils import *
import random

random.seed(2137)

img_path = get_processed_image_path(*get_random_image_id('data'))


img = cv.imread(img_path)
img.shape
```




    (320, 391, 3)


