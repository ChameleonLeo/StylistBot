from images import Images
from main import dp, bot

from aiogram import types

images = {}

menu = types.InlineKeyboardMarkup()
menu.add(types.InlineKeyboardButton("See recommendations", callback_data='recommendation'))
menu.add(types.InlineKeyboardButton("Try on default set", callback_data='default'))
# menu.add(types.InlineKeyboardButton("See style image examples", callback_data='example'))
menu.add(types.InlineKeyboardButton("Let's get started styling!", callback_data='styling'))

back_to_menu = types.InlineKeyboardMarkup()
back_to_menu.add(types.InlineKeyboardButton("Menu", callback_data='show_menu'))

upload_button = types.InlineKeyboardMarkup()
upload_button.add(types.InlineKeyboardButton('Upload images', callback_data='upload'))
upload_button.add(types.InlineKeyboardButton('Back to menu', callback_data='show_menu'))

style_button = types.InlineKeyboardMarkup()
style_button.add(types.InlineKeyboardButton("Style it, please!", callback_data='style_it'))
style_button.add(types.InlineKeyboardButton("Discard uploaded images", callback_data='discard'))
style_button.add(types.InlineKeyboardButton("Back to menu", callback_data='show_menu'))


@dp.message_handler(commands=['start'])
async def greeting(message: types.Message):
    await message.reply("Hi! I'm ChameleonStylistBot!\nPowered by aiogram.\nInspired by DLS.\n"
                        "\nI'll style your image by another image with conspicuous style.\n"
                        "You can see menu by command '/help'. Or let's start just now!", reply_markup=upload_button)
    images[message.chat.id] = Images()


@dp.message_handler(commands=['help'])
@dp.message_handler(content_types=['text'])
async def show_menu(message: types.Message):
    await message.reply("That's what we can do:", reply_markup=menu)


@dp.callback_query_handler(lambda button: button.data == 'recommendation')
async def show_recommendations(message: types.Message):
    await message.reply("While styling image will be cropped, so ...", reply_markup=back_to_menu)


@dp.callback_query_handler(lambda button: button.data == 'styling')
async def styling(callback_query):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("push the button below to make me able to receive your images.")
    await callback_query.message.edit_reply_markup(reply_markup=upload_button)


@dp.callback_query_handler(lambda button: button.data == 'default')
async def style_default_set(callback_query):
    await bot.answer_callback_query(callback_query.id)
    images[callback_query.from_user.id].default_set()

    await callback_query.message.edit_text("I set this default images (see above), let's start?")
    await callback_query.message.edit_reply_markup(reply_markup=style_button)


@dp.callback_query_handler(lambda button: button.data == 'upload')
async def style_default_set(callback_query):
    await bot.answer_callback_query(callback_query.id)
    images[callback_query.from_user.id].content_image = 0
    images[callback_query.from_user.id].style_image = 0

    await callback_query.message.edit_text("I got your images, let's start?")
    await callback_query.message.edit_reply_markup(reply_markup=style_button)


@dp.callback_query_handler(lambda button: button.data == 'discard')
async def style_default_set(callback_query):
    await bot.answer_callback_query(callback_query.id)
    images[callback_query.from_user.id].content_image = 0
    images[callback_query.from_user.id].style_image = 0
    await callback_query.message.edit_text("Images discarded, let's upload again?")
    await callback_query.message.edit_reply_markup(reply_markup=upload_button)


@dp.message_handler(content_types=['photo', 'document'])
async def upload_images(message: types.Message):
    try:
        image_id = await bot.get_file(message.photo[-1])
        image = await bot.download_file(image_id.file_path)
        images[message.chat.id].content_image = image
    except Exception as ex:
        await bot.send_message('DEBUG_ID', "Error occured: " + str(ex))
        await message.reply("Something went wrong. Please, try later.", reply_markup=upload_button)

    if message.chat.id not in images:
        await message.reply("Please, push the button below to make me able to receive your images.",
                            reply_markup=upload_button)
        return

    await message.reply("That's it! Let's style something else?", reply_markup=upload_button)
