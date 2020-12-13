# MRI Tumor Identifier App

Used transfer learning to classify between brain MRI images that had a tumor present and those that did not. Used InceptionV3 model to train over 120 epochs with little over 250 MRI images to achieve an 88% accuracy on test data. The number of epochs needed was derived from visual inspection, i.e. plotting accuracy and loss for validation and training sets over said amount of epochs. 

Using MXNet's gluon API I was able to create the model, test it, save the model's parameters and load them when needed to provide a prediction. This served useful when building the simple web-app. Unfortunately, I was not able to continue the project therefore the app only accepts one image at a time. Of course, it would be practical if it could accept a plethora of images within a directory. 

