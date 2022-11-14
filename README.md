# INFORMATICS-8000

Before the project:
Individual shapes were extracted from microscopic images which were manually segmented in Ilastik. These shapes are in the folders C (control), NS (N-Stress) and PS (P-Stress) respectively in the Data Acquisition folder.

Project accomplishments:
1.Data Acquisition:

The process is divided into two codes: Pre-ICP (FinalPipeline_PreICP.py) & ICP (FinalPipeline_ICP.py)

ICP is the Iterative Closest Point algorithm and I have implemented the following package in Python 3.6:  https://pypi.org/project/ICP/
 

o FinalPipeline_PreICP.py:

This code first calculates the perimeter and area of a shape and stores it in 101_AP.csv.
Then, it does pre-alignment of a shape to align it relative to the model shape which is the pre-processing step before ICP (to avoid getting stuck at the local minima during ICP). The pre-alignment includes scaling such that centroid size equals one, translation such that the centroid is at the origin followed by rotation to align the shape as best as I could with the model shape. Then the image is cropped to remove the axes.

This code doesn’t need ICP yet. If you run the code I have provided, then we are aligning 101.jpg (from NS) with the model shape. All folders should be accessible as they are at the same location as the code.

o FinalPipeline_ICP.py:

To run this code, ICP needs to be downloaded and set up in your system. We also need to modify the ICP.py file in the ICP downloaded package which is explained below:

As the ICP does not provide the superimposed coordinates as a result, we changed the original ICP code (where it is stored in your system) such that the data, model and superimposed coordinates get stored in NS folder as datacoordinates.txt, modelcoordinates.txt & superimposedcoordinates.txt. The modified ICP.py is also given in the folder.
(We are also storing rotation and translation matrix as well although it is not being used later.)
This code runs the ICP superimposing the cropped, pre-aligned shape (101_pre_crop.jpg) followed by the calculation of ‘Hausdorf distance’ and ‘Frechet distance’. These get stored in 101_HF.csv (I ran the code for model.jpg before).
