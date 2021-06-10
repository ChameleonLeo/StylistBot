import st_model
from images import Images
from main import dp, bot, executor

from aiogram import types

images = {}

menu = types.InlineKeyboardMarkup()
menu.add(types.InlineKeyboardButton("See recommendations", callback_data='recommendation'))
menu.add(types.InlineKeyboardButton("Try on default set", callback_data='default_set'))
# menu.add(types.InlineKeyboardButton("See style image examples", callback_data='example'))
menu.add(types.InlineKeyboardButton("Let's get started styling!", callback_data='upload'))

back_to_menu = types.InlineKeyboardMarkup()
back_to_menu.add(types.InlineKeyboardButton("Menu", callback_data='show_menu'))

upload_button = types.InlineKeyboardMarkup()
upload_button.add(types.InlineKeyboardButton('Upload images', callback_data='upload'))
upload_button.add(types.InlineKeyboardButton('Menu', callback_data='show_menu'))

style_button = types.InlineKeyboardMarkup()
style_button.add(types.InlineKeyboardButton("Style it!", callback_data='style_it'))
style_button.add(types.InlineKeyboardButton("Discard uploaded images", callback_data='discard'))
style_button.add(types.InlineKeyboardButton("Back to menu", callback_data='show_menu'))


@dp.message_handler(commands=['start'])
async def greeting(message: types.Message):
    await message.reply("Hi! I'm ChameleonStylistBot!\nPowered by aiogram.\nInspired by DLS.\n"
                        "\nI'll style your image by another image with conspicuous style.\n"
                        "You can see menu by command '/help' or Menu button.")
    await bot.send_message(message.chat.id,
                           "Or let's start just now!\nPush 'Upload images' to make me able to receive your images.",
                           reply_markup=upload_button)
    images[message.chat.id] = Images()


@dp.message_handler(commands=['help'])
@dp.message_handler(content_types=['text'])
async def show_menu(message: types.Message):
    await message.reply("That's what we can do:", reply_markup=menu)


@dp.callback_query_handler(lambda button: button.data == 'show_menu')
async def show_menu(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("That's what we can do:")
    await callback_query.message.edit_reply_markup(reply_markup=menu)


@dp.callback_query_handler(lambda button: button.data == 'recommendation')
async def show_recommendations(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Here are some recommendations, that may be useful:\n\n"
                                           "1. I can recognize only standart image formats: jpeg, png, bmp.\n"
                                           "2. Please, don't upload too big images (5 Mb max).\n"
                                           "3. Note that bigger size of images raise the time of process.\n"
                                           "4. While styling image will be cropped (output has square form),"
                                           "so make sure that main content has center position on your image.\n"
                                           "5. Style image should has distinct style. Best variant is some painting.\n"
                                           "6. I'll better work with landscape or still life images.")
    await callback_query.message.edit_reply_markup(reply_markup=back_to_menu)


@dp.callback_query_handler(lambda button: button.data == 'default_set')
async def style_default_set(callback_query: types.CallbackQuery):
    if callback_query.message.chat.id not in images:
        images[callback_query.message.chat.id] = Images()
    else:
        images[callback_query.message.chat.id].default_set()
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Set default images:")
    def_cnt = open('images/content_img.jpg', 'rb')
    """images[callback_query.message.chat.id].content_image = def_cnt
    print(def_cnt)"""
    await bot.send_photo(callback_query.message.chat.id, def_cnt)
    def_st = open('images/style_img.jpg', 'rb')
    """images[callback_query.message.chat.id].style_image = def_st
    print(def_st)"""
    await bot.send_photo(callback_query.message.chat.id, def_st)
    await bot.send_message(callback_query.message.chat.id,
                           "I set this default images, let's style?",
                           reply_markup=style_button)


@dp.callback_query_handler(lambda button: button.data == 'upload')
async def style_default_set(callback_query: types.CallbackQuery):
    if callback_query.message.chat.id not in images:
        images[callback_query.message.chat.id] = Images()
    else:
        images[callback_query.message.chat.id].content_image = 0
        images[callback_query.message.chat.id].style_image = 0
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("I'm ready to get your content image")
    await callback_query.message.edit_reply_markup(reply_markup=back_to_menu)


@dp.callback_query_handler(lambda button: button.data == 'discard')
async def discard_images(callback_query: types.CallbackQuery):
    images[callback_query.message.chat.id].content_image = 0
    images[callback_query.message.chat.id].style_image = 0
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Images discarded, let's upload something else?")
    await callback_query.message.edit_reply_markup(reply_markup=upload_button)


@dp.message_handler(content_types=['photo', 'document'])
async def upload_images(message: types.Message):
    try:
        if message.chat.id not in images:
            await message.reply("Please, firstly push 'Upload images' to make me able to receive your images.",
                                reply_markup=upload_button)
            return
        if message.content_type == 'photo':
            file = message.photo[-1]
        elif message.content_type == 'document':
            file = message.document
        else:
            await message.reply("Unrecognized format of file. Please provide me with image.",
                                reply_markup=upload_button)
            return
        image_id = await bot.get_file(file.file_id)
        image = await bot.download_file(image_id.file_path)
        if images[message.chat.id].content_image == 0:
            images[message.chat.id].content_image = image
        else:
            images[message.chat.id].style_image = image
    except Exception as ex:
        await bot.send_message('1785768192', "Error occured: " + str(ex))
        await message.reply("Something went wrong. Please, check recommendations and try later.", reply_markup=menu)

    if images[message.chat.id].content_image != 0 and images[message.chat.id].style_image == 0:
        await message.reply("Nice. Now upload style image")

    elif images[message.chat.id].content_image != 0 and images[message.chat.id].style_image != 0:
        await bot.send_message(message.chat.id,
                               "Cool! Now I'm ready to style your content image with style image.",
                               reply_markup=style_button)


@dp.callback_query_handler(lambda button: button.data == 'style_it')
async def styling(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("I started the process. It may take few minutes.\n"
                                           "Actually, you can close the chat, "
                                           "I'll send you a message when it get ready.")
    try:
        styled = await st_model.run_style_transfer(images[callback_query.message.chat.id].content_image,
                                                   images[callback_query.message.chat.id].style_image)
        await bot.send_photo(callback_query.message.chat.id, styled)
    except Exception as ex:
        await bot.send_message('1785768192', "Error occured: " + str(ex))
        await callback_query.message.reply("Something went wrong. Please, try later.")

    await bot.send_message(callback_query.message.chat.id,
                           "That's it! Let's style something else?",
                           reply_markup=upload_button)
    del images[callback_query.message.chat.id]


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
