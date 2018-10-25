import telepot
from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np


class TelegramCallback(Callback):

    def __init__(self, config, plot_metrics=True, temp_image_path='temp.jpg', plot_n_epochs=100):
        super(TelegramCallback, self).__init__()
        self.chat_id = config['chat_id']
        self.bot = telepot.Bot(config['token'])
        self.img_path = temp_image_path
        
        self.plot_metrics = plot_metrics
        self.plot_n_epochs = plot_n_epochs
        
        self.track_vals = {}
        self.epochs = []
        
    def set_proxy(self, proxy_config):
        if "username" in proxy_config:
            basic_auth = (proxy_config['username'],  proxy_config['password'])
            telepot.api.set_proxy(proxy_config['proxy_url'], basic_auth)
        else:
            telepot.api.set_proxy(proxy_config['proxy_url'])
    
    def update_track_vals(self, epoch, logs={}): 
        self.epochs.append(epoch), 
        for k, v in logs.items():
            self.track_vals[k].append(v)
    
    def plot_and_save_graph(self):
        n_plts = len(self.model.metrics_names)
        plt.figure(figsize = (10, n_plts * 5))
        for i, k in enumerate(self.model.metrics_names):
            
            train_max, train_min = np.max(self.track_vals[k]), np.min(self.track_vals[k])
            val_max, val_min = np.max(self.track_vals['val_' + k]), np.min(self.track_vals['val_' + k])

            plt.subplot(n_plts, 1, i + 1)
            plt.title(k + " - max: {:.5f}; min: {:.5f}".format(train_max, train_min) + "\nval_" + \
                      k + " - max: {:.5f}; min: {:.5f}".format(val_max, val_min))

            plt.plot(self.epochs, self.track_vals[k], '.-', color = 'b')    
            plt.plot(self.epochs, self.track_vals['val_' + k], '.-', color = 'r')

            # plot min max points
            plt.plot([np.argmin(self.track_vals[k]), np.argmax(self.track_vals[k])],
                     [train_min, train_max], 'b*', markersize = 12)
            plt.plot([np.argmin(self.track_vals['val_' + k]), np.argmax(self.track_vals['val_' + k])],
                     [val_min, val_max], 'r*', markersize = 12)

            plt.legend([k, 'val_' + k])
            plt.grid()
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
        if epoch == 0:
            for k in self.model.metrics_names:
                self.track_vals[k] = []
                self.track_vals['val_' + k] = []
                
        for k in self.model.metrics_names:
            text += '{}: {:.4f}; {}: {:.4f};\n'.format(k, logs[k], 'val_' + k, logs['val_' + k])
        
        #for k, v in logs.items():
        #    text += '{}: {:.4f};'.format(k, v)
        
        self.update_track_vals(epoch, logs)
        self.send_message(text)
        
        if self.plot_metrics and (epoch + 1) % self.plot_n_epochs == 0:
            self.plot_and_save_graph()
            self.send_image(self.img_path)
     
    def on_train_end(self, logs={}):
        self.plot_and_save_graph()
        if self.plot_metrics:
            self.send_image(self.img_path)