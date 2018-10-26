import telepot
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

class TelegramCallback(Callback):

    def __init__(self, config, plot_metrics=True, temp_image_path='temp.jpg', plot_n_epochs=100, 
                figsize=(10, 5)):
        
        super(TelegramCallback, self).__init__()
        
        self.chat_id = config['chat_id']
        self.bot = telepot.Bot(config['token'])
        
        self.plot_metrics = plot_metrics
        self.plot_n_epochs = plot_n_epochs
        self.img_path = temp_image_path
        self.figsize = figsize
        
        self.log = {}
        self.epochs = []
        self.lr_on = False
        self.valid_on = False
        
    def set_proxy(self, proxy_config):
        if "username" in proxy_config:
            basic_auth = (proxy_config['username'],  proxy_config['password'])
            telepot.api.set_proxy(proxy_config['proxy_url'], basic_auth)
        else:
            telepot.api.set_proxy(proxy_config['proxy_url'])
    
    def add_log(self, epoch, logs={}): 
        self.epochs.append(epoch)
        for k, v in logs.items():
            self.log.setdefault(k, [])
            self.log[k].append(v)
    
    def plot_and_save_graph(self, metric):
        fig, ax = plt.subplots(figsize=self.figsize)
        
        title = metric
        legend = [metric]
        if metric == 'loss':
            star_point = [np.argmin(self.log[metric]) , np.min(self.log[metric])]
            title += " - min: {:.5f};".format(star_point[1]) + "\n"
            legend.append(metric + ' min')
        else:
            star_point = [np.argmax(self.log[metric]) , np.max(self.log[metric])]
            title += " - max: {:.5f};".format(star_point[1]) + "\n"
            legend.append(metric + ' max')
        
        ax.plot(self.epochs, self.log[metric], '.-', color = 'b')    
        ax.plot(star_point[0], star_point[1], 'b*', markersize = 12)
        ax.set_xlabel('epochs')

        if self.valid_on:
            val_metric = 'val_' + metric
            title += val_metric
            legend.append(val_metric)
            if val_metric == 'val_loss':
                star_point = [np.argmin(self.log[val_metric]) , np.min(self.log[val_metric])]
                title += " - min: {:.5f};".format(star_point[1])
                legend.append(val_metric + ' min')
            else:
                star_point = [np.argmax(self.log[val_metric]) , np.max(self.log[val_metric])]
                title += " - max: {:.5f};".format(star_point[1])
                legend.append(val_metric + ' max')
            
            ax.plot(self.epochs, self.log[val_metric], '.-', color = 'r')
            ax.plot(star_point[0], star_point[1], 'r*', markersize = 12)

        if self.lr_on:
            ax2 = ax.twinx()
            ax2.plot(self.epochs, self.log['lr'], color = 'green')
            ax2.set_ylabel('lr',  color='green')

        ax.set_title(title)
        ax.legend(legend)
        ax.grid()
        fig.tight_layout()
        plt.savefig(self.img_path)
    
    def send_message(self, text):
        try:
            self.bot.sendMessage(self.chat_id, text)
        except Exception as e:
            print('Message did not send. Error: {}.'.format(e))

    def send_image(self, img_path):
        try:
            self.bot.sendPhoto(self.chat_id, open(img_path, 'rb'))
        except Exception as e:
            self.send_message('Image did not send. Error: {}.'.format(e))
            print('Image did not send. Error: {}.'.format(e))
    
    def on_train_begin(self, logs={}):
        text = 'Start training model {}.'.format(self.model.name)        
        self.send_message(text)

    def on_epoch_end(self, epoch, logs={}):
        text = 'Epoch {}.\n'.format(epoch)
        for k, v in logs.items():
            text += '{}: {:.4f};\n'.format(k, v)

        self.add_log(epoch, logs)
        self.send_message(text)
        
        if self.plot_metrics and (epoch + 1) % self.plot_n_epochs == 0:
            self.plot_and_save_graph()
            self.send_image(self.img_path)
     
    def on_train_end(self, logs={}):  
        self.lr_on = 'lr' in self.log.keys()
        self.valid_on = 'val_loss' in self.log.keys()        
        if self.plot_metrics:
            for metric in self.model.metrics_names:
                self.plot_and_save_graph(metric)
                self.send_image(self.img_path)