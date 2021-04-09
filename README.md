# Beer Price Checker
Take a photo, find the best price at a grocer near you

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

![alt text](https://github.com/azwinlam/UberEats-Analysis/blob/main/figures/cuisineCounts.png

## Modelling
* The baseline CCN model was taken from https://www.tensorflow.org/tutorials/images/classification.
* Validation accuracy improved significantly when dropout layers were added to the model. 
