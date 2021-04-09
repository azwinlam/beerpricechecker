![alt text](https://github.com/azwinlam/beerpricechecker/blob/main/images/title.png "Title")

# Beer Price Checker
Take a photo, find the best price at a grocer near you
* http://tinyurl.com/beerpricechecker
* Streamlit server hosted on Google Cloud Platform VM. Should be running until June 30, 2021.

## Objective
* To find the best price for a beer you wish to purchase at a grocer near you.

## Project Overview
* Scraped over 3500+ images for 18 brands of beer.
* Cropped images to train Convolutional Neural Network
* Identifed beer brands with high accuracy and provided best price for purchase

## Code and Resources
**Python Version:** Python 3.7.10 (Google Colab)

**Tensorflow** 2.4.1

**Opencv-python** 4.1.2

**Inbac** 2.1.0
 
**Packages:** pandas, numpy, matplotlib, seaborn, selenium, 

**Teammates**: [Alex Li's GitHub](https://github.com/ahhhlexli "Alex Li's GitHub") & [lhwj0619's GitHub](https://github.com/lhwj0619 "lhwj0619's GitHub")

## Web Scraping
* Initially started with manual scraping of beer logo images to test the viability of the image recognition project.
* After some promising results from the CNN, we used some template python codes to scrape images off Google Search.
* Selenium was used to scrape the price information from https://online-price-watch.consumer.org.hk/opw/search/asahi

## Data Cleaning
* The data from Consumer Council was placed in a pandas DataFrame. Had to remove minor data figures for formatting.
* Images were cropped using a standalone python .exe called inbac.
* Labels were applied to the images using folders.

![alt text](https://github.com/azwinlam/beerpricechecker/blob/main/images/consumercouncil.png "Consumer Council")

## Modelling
* The baseline CCN model was taken from https://www.tensorflow.org/tutorials/images/classification.
* Validation accuracy improved significantly when dropout layers were added to the model. 

**Baseline Model**
```
model = tf.keras.Sequential([
  layers.experimental.preprocessing.Rescaling(1./255),
  #layers.Conv2D(input_shape=(img_height,img_width,3),filters=64,kernel_size=(2,2),padding="same", activation="relu"),
  layers.Conv2D(64, 3, activation='relu'),
  layers.Conv2D(64, 2, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 2, activation='relu'),
  layers.Conv2D(128, 2, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes, activation='softmax')
])
```

**Final Model**
```
model = tf.keras.Sequential([data_augmentation])
model.add(Conv2D(input_shape=(img_height,img_width,3),filters=64,kernel_size=(3,3),padding="same", activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(MaxPooling2D(pool_size=2,)) 
model.add(Dropout(0.2))
model.add(Conv2D(kernel_size = 2, filters = 64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(Conv2D(kernel_size = 2, filters = 64, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(kernel_size = 2, filters = 128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(Conv2D(kernel_size = 2, filters = 128, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv2D(kernel_size = 2, filters = 256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(Conv2D(kernel_size = 2, filters = 256, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
model.add(MaxPooling2D(pool_size = 2))
model.add(Dropout(0.2))
model.add(GlobalMaxPooling2D())
#model.add(GlobalAveragePooling2D())
model.add(Dense(num_classes, activation = 'softmax', kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)))
```

## Production
* Streamlit was used to deploy the model. 

## Presentation
[PowerPoint](https://github.com/azwinlam/beerpricechecker/blob/main/Beer%20Price%20Checker.pptx)
