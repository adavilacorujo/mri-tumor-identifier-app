import mxnet as mx 
import numpy as np 
import gluoncv
import matplotlib.pyplot as plt 

from mxnet import gluon, nd, autograd

"""
This class uses mxnet's gluon api to use transfer learning for the task at hand, i.e. classifying 
if an MRI image of a brain presents a tumor or not. 
The class relies heavily on MXNet's API from simple nd arrays to optimizers
"""

class MRI():
    """
    Function takes as input two data loaders. Both of them are required to be part of MXNet's mx.gluon.data.Dataloader() class.
    The rest are left at the users discretion. 
    """
    def __init__(self, 
            train_data_loader   = None,             
            val_data_loader     = None,
            loss                = gluon.loss.SigmoidBinaryCrossEntropyLoss(),
            optimizer           = 'sgd',
            model_name          = 'inceptionv3',
            metric              = mx.metric.Accuracy()):

        self.train_data_loader  = train_data_loader
        self.val_data_loader    = val_data_loader
        self.L = loss
        self.optimizer = optimizer 
        self.metric = metric
        self.ctx = mx.gpu() if mx.context.num_gpus() else mx.cpu()

        self.model = self.define_model(model_name)


    """
    Train method, trains the model on batches found in each Dataloader. 
        
        Input:
            - EPOCHS: 'int'         number of epochs to perform forward and back propagation
            - params: 'dictionary'  parameters for the optimizer. If parameters do not match script will produce an error. 
            - plot:   'bool'        boolean value. If True plots validation and train loss and accuracy over the number of 
                                epochs provided. Used to determine correct number of epochs through visual inspection. 

    """
    def train(self, EPOCHS, params = {'learning_rate': 0.001, 'momemtum': 0.005}, plot=False):
        batch_size = 0
        trainer = gluon.Trainer(self.model.collect_params(), self.optimizer, params)

    
        training_loss, validation_loss, \
        training_accuracy, validation_accuracy = list(),list(), list(), list()    

        for epoch in range(EPOCHS):
            self.metric.reset()
            train_loss, val_loss = 0, 0
            
            """
            Training loop
            """
            for i, (data, label) in enumerate(self.train_data_loader):
                data    = data.as_in_context(self.ctx).astype('float32')
                label   = label.as_in_context(self.ctx).astype('float32')                    

                with autograd.record():
                    output = self.model(data)
                    last_layer = gluon.nn.Activation('sigmoid')(output)
                    predictions = nd.array([1 if out > 0.5 else 0 for out in last_layer]).astype('float32')
                    loss = self.L(output, label)
                
                loss.backward()
                trainer.step(data.shape[0])
                train_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                self.metric.update(label, predictions)

                if i == (len(data) - 1):
                    continue
                else:
                    batch_size = data.shape[0]

            
            _, train_accuracy = self.metric.get()
            train_loss /= batch_size

            self.metric.reset()

            """
            Validation loop
            """
            for i, (data, label) in enumerate(self.val_data_loader):
                data    = data.as_in_context(self.ctx).astype('float32')
                label   = label.as_in_context(self.ctx).astype('float32')                    

                with autograd.record():
                    output = self.model(data)
                    last_layer = gluon.nn.Activation('sigmoid')(output)
                    predictions = nd.array([1 if out > 0.5 else 0 for out in last_layer]).astype('float32')
                    loss = self.L(output, label)
                
                val_loss += sum([l.mean().asscalar() for l in loss]) / len(loss)
                self.metric.update(label, predictions)

                if i == (len(data) - 1):
                    continue
                else:
                    batch_size = data.shape[0]

        
            _, val_acc = self.metric.get()
            val_loss /= batch_size

            print("Epoch {}, val_loss: {:.3f}, train_loss {:.3f}, val_acc {:.2f}, train_acc: {:.2f}".format(epoch, val_loss, train_loss, val_acc, train_accuracy))
            if plot:
                validation_accuracy.append(val_acc)
                training_accuracy.append(train_accuracy)
                validation_loss.append(val_loss)
                training_loss.append(train_loss)
            else:
                training_loss, validation_loss, \
                training_accuracy, validation_accuracy = None, None, None, None
                
        if plot:
            data = [[(validation_accuracy, 'Val Acc'), (training_accuracy, 'Train Acc')], \
                [(validation_loss,'Val Loss'), (training_loss, 'Train Loss')]]
            for nested in data:
                plt.figure()
                for d, title in nested:
                    plt.plot(range(epoch + 1), d, label=title)
                    plt.legend()
            plt.show()

        trainer = None    
        training_loss, validation_loss, \
                training_accuracy, validation_accuracy = None, None, None, None
            
    """
    Method to define model. Model is determined by user's preference when initiating class instance

        Input:
            - model_name:   'str' parameter to select a model from MXNet's plethora of available models 
                for transfer learning.
        
        Output:
            - model:        'gluon.HybridBlock' pre-defined model by name
    """
    def define_model(self, model_name):
        model = gluoncv.model_zoo.get_model(model_name, pretrained=True, ctx=self.ctx)
        with model.name_scope():
            model.output = gluon.nn.Dense(1)
        model.output.initialize(mx.init.Xavier(), ctx=self.ctx)
        model.hybridize()
        return model

    """
    Method to get accuracy for any data_loader provided. 
        Input:
            - data_loader:  'mx.gluon.data.Dataloader'  data loader containing batches of images of shape (c, h, w) and its 
                        corresponding labels

        Output:
            - accuracy:     'float' accuracy of the model on dataloader provided 
    """
    def get_accuracy(self, data_loader):
        predictions = list()
        labels = list()
        for data, l in data_loader:
            data = data.as_in_context(self.ctx).astype('float32')

            output = self.model(data)
            last_layer = gluon.nn.Activation('sigmoid')(output)
            predictions.append(nd.array([1 if out > 0.5 else 0 for out in last_layer]).astype('float32'))
            labels.append(l)
        predictions = np.ndarray.flatten(np.array(predictions))
        preds = list()
        for pred in predictions:
            preds.append(pred.asnumpy()[0])
        accuracy = (1 - (np.mean(preds  != list(labels)[0].asnumpy().astype('float32'))))
        return accuracy

    """
    Method to get predictions of the model. Provides users with a tuple containing the images predictions along with its
    confidence interval. 
        Input:
            - data_loader:  'mx.gluon.data.Dataloader'  data loader containing batches of images of shape (c, h, w) and its 
                        corresponding labels
        Output:
            - return_val:   'tuple' tuple of size 3, ([img], [prediction], [conf_interval]), where each dimension contains an 
                        array of images, predictions and its corresponding confidence interval
    """
    def get_predictions(self, data_loader):
        """
        Returns (images, prediction and confidence interval) 
        """
        if isinstance(data_loader, mx.gluon.data.DataLoader):
            data            = list()
            confidence_int  = list()
            predictions     = list()
            for i, (imgs, _) in enumerate(data_loader):
                
                imgs = imgs.as_in_context(self.ctx).astype('float32')
                output = self.model(imgs)
                last_layer = gluon.nn.Activation('sigmoid')(output)
                predictions.append(nd.array([1 if out > 0.5 else 0 for out in last_layer]).astype('float32'))

                data.append(imgs)
                confidence_int.append(last_layer)
                
            predictions = ['Detected' if pred == 1 else 'Not detected' for pred in predictions[0]]
            return_val = (  
                    data[0], 
                    confidence_int[0],
                    predictions
                   )
            return return_val
        

    """
    Method to visualize predictions made by the model
    Input:
        - data: 'tuple' tuple of size 3, ([img], [prediction], [conf_interval]), where each dimension contains an 
                        array of images, predictions and its corresponding confidence interval
    """
    def visualize(self, data):
        for i in range(len(data[2])):
            img         = data[0][i]
            interval    = data[1][i].asscalar()
            prediction  = data[2][i]
            
            if 'Not' in prediction:
                interval = 1 - interval
            img = nd.transpose(data=img, axes=(1,2,0))
            self.plot_mx_array(img, prediction, interval*100)

    """
    Method to plot images as mx arrays. 

    To be called by visualize, ideally. 

    Input:
        - array:    'list'  list of size HxWxC containing an image.
        - pred:     'str'   Either "Not Detected" or "Detected", used to set title of plot.
        - interval: 'float' Confidence interval, used as title as well. 
    """
    def plot_mx_array(self, array, pred, interval):
        assert array.shape[2] == 3
        title = "{}: {:.2f}% Confidence".format(pred, interval)
        plt.imshow((array.clip(0, 255)/ 255).asnumpy())
        plt.title(title)
        plt.show()

    """
    Helper function to save model's parameters.
    Input:
        - file_name:    'str' directory and filename of future file to hold parameters
    """
    def save_params(self, file_name):
        self.model.save_parameters(file_name)


    """
    Helper function to get model's parameters.
    Input:
        - file_name:    'str' directory and filename of future file to hold parameters
    """
    def load_params(self, file_name):
        self.model.load_parameters(file_name)