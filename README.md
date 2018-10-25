## Keras callback in Telegram
Telegram-bot (telepot) callback for your Keras model

## For callback you need to get `token` and `chat_id`

#### Register telegram bot and get `token`.  
 - Find `BotFather` in Telegram.
 - Follow `BotFather` instructions to register your bot in a few steps and get `token`.

#### Get `chat_id` between you and bot.
 - Open the chat with your new bot and write to him /start and one more message
 - Run this code

```python
import telepot
# TELEGRAM_BOT_TOKEN = 533254114:AAF9I43PyjHe5uFv2dfPdqBPy4cxofbDnGM
bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
bot.getUpdates()[0]['message']['chat']['id']
```

If you telegram blocked in your country use proxy
```python
import telepot
bot = telepot.Bot(TELEGRAM_BOT_TOKEN)
RROXY_URL = 'http://10.10.10.10:1080' # paste your proxy
telepot.api.set_proxy(RROXY_URL)
# telepot.api.set_proxy(RROXY_URL, ('username', 'password')) 
bot.getUpdates()[0]['message']['chat']['id']
```

### Example

```python
from .callbacks import TelegramCallback

...
# create callback
config = {
    'token': '11111111:xxxxxxxxxxxxxxxxxxxxxxxxxxxx',   # paste your bot token
    'chat_id': 123456789,                               # paste your chat_id
}
tg_callback = TelegramCallback(config)

# add proxy if it necessary
proxy_config = {
    'proxy_url':'http://10.10.10.10:1080',
    'username':'',
    'password':''
}
tg_callback.set_proxy(proxy_config)

# start training
model.fit(x, y, batch_size=32, callbacks=[tg_callback])
```
