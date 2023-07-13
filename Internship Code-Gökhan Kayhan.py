#Reference = https://github.com/bnsreenu/python_for_microscopists

import numpy as np
import cv2
import pandas as pd

import pickle
import os
from matplotlib import pyplot as plt

from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.filters.rank import entropy, gradient, mean
from skimage.morphology import disk

#Preprocessing
img_name=' '

img = cv2.imread(img_name)
median = cv2.medianBlur(img, 3)

x=92 #These coordinates are used to remove
w=328 # the black areas at the edge by cropping the image.
y=0
h=512

dim=(512,512)
img=cv2.resize(img,dim) #image size is reduced to 512 x 512
median=cv2.resize(median,dim) # Median filter is applied to image

imgray = cv2.cvtColor(median,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(imgray,50,255,cv2.THRESH_BINARY) #Image is converted into binary(black-white) format

contours, _ = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)#the contours are found
for cnt in contours:
    area=cv2.contourArea(cnt) # For each contour are is calculated
    print(area)
    if area<15000 and area>1000: #If the area is between certain values that means
        cv2.drawContours(img, cnt, -1, (0,255,0), 3) #it is one of the unnecessary parts in the picture.
        
        clean=cv2.fillConvexPoly(img, cnt, 0)#These areas are removed by filling them with black
        clean = cv2.medianBlur(clean, 5) #median filter is applied again on the cleaned image
        crop_image = clean[ y:y+h , x:x+w]#The black bands on the right and left are removed from the image
        cv2.imwrite(img_name +".jpg", crop_image)#Final image is saved

###



image_dataset=pd.DataFrame()# dataframe to capture image features

def feature_extraction(img):
    
        pixel_values = img.reshape(-1) # Store pixel values into a single column
        df['Pixel=Value'] = pixel_values # pixel value itself as a feature
        df['Image_Name']=image
        
        #Gabor features
        num = 1
        kernels = []
        for theta in range(2): 
            theta = theta / 4. * np.pi
            for sigma in (1, 3):
                for lamda in np.arange(0, np.pi, np.pi / 4): 
                    for gamma in (0.05, 0.5):
                                      
                        gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
                        ksize=9
                        kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                        kernels.append(kernel)
                         
                        fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)#apply filter to image
                        filtered_img = fimg.reshape(-1)
                        
                        df[gabor_label] = filtered_img
                        num += 1  #Increment for gabor column label

        #OTHER FEATURES
        #Entropy
        ent = entropy(img, disk(5))
        ent1 = ent.reshape(-1)
        df['Entropy'] = ent1

        #Gradient
        grad = gradient(img, disk(5))
        grad1 = grad.reshape(-1)
        df['Gradient'] = grad1

        #Mean
        avg = mean(img, disk(5))
        avg1 = avg.reshape(-1)
        df['Average'] = avg1
              
        #CANNY EDGE
        edges = cv2.Canny(img, 100,200)  
        edges1 = edges.reshape(-1)
        df['Canny Edge'] = edges1

        #ROBERTS EDGE
        edge_roberts = roberts(img)
        edge_roberts1 = edge_roberts.reshape(-1)
        df['Roberts'] = edge_roberts1

        #SOBEL
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1

        #GAUSSIAN with sigma=3
        from scipy import ndimage as nd
        gaussian_img = nd.gaussian_filter(img, sigma=3)
        gaussian_img1 = gaussian_img.reshape(-1)
        df['Gaussian s3'] = gaussian_img1

        #GAUSSIAN with sigma=7
        gaussian_img2 = nd.gaussian_filter(img, sigma=7)
        gaussian_img3 = gaussian_img2.reshape(-1)
        df['Gaussian s7'] = gaussian_img3

        #MEDIAN with sigma=3
        median_img = nd.median_filter(img, size=3)
        median_img1 = median_img.reshape(-1)
        df['Median s3'] = median_img1

        #VARIANCE with size=3
        variance_img = nd.generic_filter(img, np.var, size=3)
        variance_img1 = variance_img.reshape(-1)
        df['Variance s3'] = variance_img1  #Add column to original dataframe
    
        
        #update data frame for images to include details for each image
        image_dataset = image_dataset.append(df)

        return image_dataset
    


img_path = "images/train_images/"
for image in sorted(os.listdir(img_path)): 
    print(image)

    df=pd.DataFrame()# Temporary data frame

    input_image=cv2.imread(img_path+image)
    img = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    
    image_dataset = feature_extraction(img)



mask_dataset=pd.DataFrame() #create dataframe to capture mask

mask_path = "images/train_masks"
for mask in sorted(os.listdir(mask_path)):
    
    df2 = pd.DataFrame()#Temporary dataframe 
    input_mask = cv2.imread(mask_path+mask)
    label = cv2.cvtColor(input_mask, cv2.COLOR_BGR2GRAY)

    label_values = label.reshape(-1)
    df2['Label_VaLue'] = label_values
    df2['Mask_Name'] = mask

    mask_dataset = mask_dataset.append(df2)
    


dataset=pd.concat([image_dataset,mask_dataset],axis=1)

X = dataset.drop(labels=["Image_Name" , "Mask_Name" , "Label_Value"] , axis=1)
#axis = 1 ; It means drop these columns

#Assign label values to Y (our prediction)
Y=dataset["Label_Value"].values

#Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)



from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 50, random_state = 42)

# Train the model on training data
model.fit(X_train, y_train)



from sklearn import metrics
prediction_test = model.predict(X_test)
print("Accuracy = ", metrics.accuracy_score(y_test, prediction_test))



#Save model
filename = "segmentation_model"
pickle.dump(model, open(filename, 'wb'))



#Segmentation Results
loaded_model = pickle.load(open(filename, 'rb'))
path = "images/Test_images/"

for image2 in os.listdir(path):
    print(image2)
    img2 = cv2.imread(path+image2)
    img2_gray = img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    Test = feature_extraction(img2_gray)
    result = loaded_model.predict(Test)
    segmented = result.reshape((img2_gray.shape))

    plt.imsave('images/Segmented/'+ image2, segmented, cmap ='jet')





