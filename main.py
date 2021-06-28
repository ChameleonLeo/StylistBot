import logging
from aiogram import types, Bot, Dispatcher
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from decouple import config


bot = Bot(token=config("API_TOKEN"))
dp = Dispatcher(bot, storage=MemoryStorage())


# Enable logging
logging.basicConfig(level=logging.WARNING,  # DEBUG
                    format=u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s] %(message)s')

# Initialize webhook and webapp settings
wh_host = f'https://{config("APP_NAME")}.herokuapp.com'
wh_path = f'/webhook/{config("API_TOKEN")}'
wh_url = f'{wh_host}{wh_path}'
wa_host = '0.0.0.0'
wa_port = int(config("PORT"))
# dp.middleware.setup(LoggingMiddleware())


async def on_startup(dispatcher):
    await bot.set_webhook(wh_url, drop_pending_updates=True)


async def on_shutdown(app):
    await bot.close()
    await dp.storage.close()
    await dp.storage.wait_closed()

# Defining of buttons for interactive communication
menu = types.InlineKeyboardMarkup()
menu.add(types.InlineKeyboardButton("See recommendations", callback_data='recommendation'))
menu.add(types.InlineKeyboardButton("Try on default set", callback_data='default_set'))
menu.add(types.InlineKeyboardButton("Let's get started styling!", callback_data='upload'))

back_to_menu = types.InlineKeyboardMarkup()
back_to_menu.add(types.InlineKeyboardButton("Back to menu", callback_data='show_menu'))

upload_button = types.InlineKeyboardMarkup()
upload_button.add(types.InlineKeyboardButton('Upload images', callback_data='upload'))
upload_button.add(types.InlineKeyboardButton('Menu', callback_data='show_menu'))

style_button = types.InlineKeyboardMarkup()
style_button.add(types.InlineKeyboardButton("Style it!", callback_data='style_it'))
style_button.add(types.InlineKeyboardButton("Discard uploaded images", callback_data='discard'))
style_button.add(types.InlineKeyboardButton("Back to menu", callback_data='show_menu'))
