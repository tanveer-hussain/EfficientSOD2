# PASNet (CVPRW2022)
## Pyramidal Attention for Saliency Detection

![Framework](https://user-images.githubusercontent.com/40714349/179899621-d2560702-7d0d-436c-9dbd-0669fabce5ff.png)

To download the framework: [saliency framework.pdf](https://github.com/tanveer-hussain/EfficientSOD2/files/9146609/saliency.framework.pdf)

### 1. Paper Links
ArXiv: https://arxiv.org/abs/2204.06788

CVPRW: https://openaccess.thecvf.com/content/CVPR2022W/FaDE-TCV/papers/Hussain_Pyramidal_Attention_for_Saliency_Detection_CVPRW_2022_paper.pdf

## 2. Setup
You need to install Pytorch (preferred 1.7.1) and some basic libraries including PIL, cv2, numpy, etc.

## 3. How to Train?

### 3.1. Datasets
The datasets can be downloaded from the following links. We follow the training and testing data similar to the previous methods.

Option 1: Download from GitHub of an existing SOD Method ([UCNet](https://github.com/JingZhang617/UCNet)): [Click here](https://drive.google.com/file/d/1zslnkJaD_8h3UjxonBz0ESEZ2eguR_Zi/view)

Option 2: Download from Google drive of existing SOD Method (D3Net): [Click here](https://github.com/DengPingFan/D3NetBenchmark/blob/master/README.md)

Option 3: Follow a dedicated GitHub page for SOD datasets: [Click here](https://github.com/GrassBro/SODdataset)

### 3.2. Training and Testing
Run train.py (current training code supports RGB-D dataset training, you can request for RGB as well or tune the code yourself)

In the training code, there is function call to the testing code which then evaluates the model's performance and stores the results in corresponding folders.

### 4. Qualitative Results
![Results](https://user-images.githubusercontent.com/40714349/179902307-22c44596-4b31-4eae-9ba4-4f0d9ee7c044.png)

### 5. Quantitative Results
*Notice:* Please follow the paper pdf to view the references.

![QuantitativeResults](https://user-images.githubusercontent.com/40714349/179902647-7fdec28e-d4c5-428f-b32c-e0ed3eb528c0.png)

## 5. Citation and Acknowledgements
Please cite our following papers on Saliency Detection if you like our work:

<pre>
<code>
@InProceedings{hussain2022Pyramidal,
    author    = {Hussain, Tanveer and Anwar, Abbas and Anwar, Saeed and Petersson, Lars and Baik, Sung Wook},
    title     = {Pyramidal Attention for Saliency Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {2878-2888}
}
}</code>
</pre>

<pre>
<code>@misc{hussain2021densely,
      title={Densely Deformable Efficient Salient Object Detection Network}, 
      author={Tanveer Hussain and Saeed Anwar and Amin Ullah and Khan Muhammad and Sung Wook Baik},
      year={2021},
      eprint={2102.06407},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}</code>
</pre>

Some of the functions in codes are inspired by [UCNet](https://github.com/JingZhang617/UCNet) GitHub repository. The authors are thankful for their nice and explained SOD GitHub page.

## 6. Contact
I would be happy to guide and assist in case of any questions and I am open to research discussions and collaboration in Saliency Detection domain. Ping me at tanveer445 [at] [ieee] [.org]




