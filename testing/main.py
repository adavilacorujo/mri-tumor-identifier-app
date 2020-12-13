import numpy as np 

from analytics.cnn import MRI
from analytics.data_ingestion import Data


if __name__ == '__main__':

    """
    Load Dataset
    """
    print("[*] Loading images...")
    
    d = Data()
    
    # Get train, val, and test images 
    x_train, y_train  = d.get_images("../data/train/", train=True)
    x_val, y_val      = d.get_images("../data/val/")
    x_test, y_test    = d.get_images("../data/test/")

    print("{} training samples".format(len(y_train)))
    print("{} validation samples".format(len(y_val)))
    print("{} testing samples".format(len(y_test)))

    # Crop training images
    x_train         = d.crop_imgs(x_train)
    x_val           = d.crop_imgs(x_val)
    x_test_cropped  = d.crop_imgs(x_test)

    # Create dataloaders for mxnet's model
    # gluon prefers data to be (num_batch, channels, n, n)
    data_shape = (3, 300, 300)
    train_dataloader    = d.data_loader(x_train, y_train, data_shape=data_shape)
    val_dataloader      = d.data_loader(x_val, y_val, data_shape=data_shape, transform=False, shuffle=True)
    test_dataloader     = d.data_loader(x_test, y_test, data_shape=data_shape, transform=False, shuffle=True)
    
    # Define model 
    net = MRI(
            train_data_loader=train_dataloader, 
            val_data_loader=val_dataloader, 
        )



    # Train model 
    # print("[*] Training...")
    # net.train(
    #     EPOCHS=120, 
    #     params={
    #         'learning_rate' : 0.001,
    #         'momentum'      : 0.005
    #     }, 
    #     plot=False)

    # # Save model 
    file_name = '../data/params/net.params'
    # net.save_params(file_name)

    # Get model from params
    net.load_params(file_name)


    test = net.get_predictions(test_dataloader)
    net.visualize(test)

    print(net.get_accuracy(test_dataloader))




    
