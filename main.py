import logging
from decouple import config
from flask import Flask
from aiogram import Bot, Dispatcher, executor


app = Flask(__name__)

API_TOKEN = config('API_TOKEN')

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format=u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s] %(message)s',
                    )

# Initialize bot and dispatcher
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
    # app.run()
