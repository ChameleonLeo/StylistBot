import logging
from decouple import config
from aiogram import Bot, Dispatcher, types


API_TOKEN = config('API_TOKEN')

logging.basicConfig(level=logging.DEBUG,
                    format=u'%(filename)s [LINE:%(lineno)d] #%(levelname)-8s [%(asctime)s] %(message)s',
                    )

bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


menu = types.InlineKeyboardMarkup()
menu.add(types.InlineKeyboardButton("See recommendations", callback_data='recommendation'))
menu.add(types.InlineKeyboardButton("Try on default set", callback_data='default_set'))
# menu.add(types.InlineKeyboardButton("See style image examples", callback_data='example'))
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
