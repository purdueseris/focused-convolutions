# focused-convolutions
Repository containing the code for the paper [Irrelevant Pixels are Everywhere: Find and Exclude Them for More Efficient Computer Vision](https://ieeexplore.ieee.org/abstract/document/9870012), published in IEEE AICAS 2022, and our upcoming _IEEE Intelligent Systems_ paper.

## What is a focused convolution?
Normal CNNs operate convolutions on the entirety of the input image. However, many input images have many pixels that are not very interesting (e.g. background pixels). This means normal CNNs are wasting time and energy on those uninteresting pixels.

The Focused Convolution is designed to ignore any pixels that are outside the Area of Interest, focusing only on interesting pixels. The weights and biases can be kept from the original CNN, allowing you to achieve the same accuracy while saving on energy and inference time.

## Installing
This can be easily done with
```bash
pip install focusedconv
```

## Demo
After installing, you can open [`focusedconv-demo.ipynb`](https://github.com/calebtung/focused-convolutions/blob/main/focused-conv-demo.ipynb) to see an example of how you can instrument VGG-16 to automatically determine which pixels should be deleted.


## Details about the source code
A separate README with more comprehensive source code details can be found in the [`src/focusedconv` directory](https://github.com/calebtung/focused-convolutions/tree/main/src/focusedconv).
