import asyncio
import threading

# from aiogram.utils import executor
from aiogram.utils.executor import start_webhook

import model
from images import Images, images, get_file_path, set_random_default_set
from utils import *


@dp.message_handler(commands=['start'])
async def greeting(message: types.Message):
    await message.reply("Hi! I'm ChameleonStylistBot!\n"
                        "\nI'll transfer texture from your style image to content image.\n"
                        "You can see menu by command /help or Menu button.")
    await bot.send_message(message.chat.id,
                           "Or let's start just now!\nPush 'Upload images' "
                           "to make me able to receive your images.",
                           reply_markup=upload_button)
    # Inserting of user's id to special dict for further handling his images
    images[message.chat.id] = Images()


@dp.message_handler(commands=['about'])
async def about(message: types.Message):
    await message.reply("ChameleonStylistBot.\nPowered by aiogram.\nInspired by DLS.", reply_markup=menu)


# Handle '\help' command or messages with random text and other content types
@dp.message_handler(commands=['help'])
@dp.message_handler(content_types=['text', 'audio', 'video', 'video_note',
                                   'voice', 'location', 'contact', 'sticker'])
async def show_menu(message: types.Message):
    await message.reply("That's what we can do:", reply_markup=menu)


# Callback for "Menu" button
@dp.callback_query_handler(lambda button: button.data == 'show_menu')
async def show_menu(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("That's what we can do:")
    await callback_query.message.edit_reply_markup(reply_markup=menu)


# Callback for "See recommendations" button
@dp.callback_query_handler(lambda button: button.data == 'recommendation')
async def show_recommendations(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Here are some recommendations, that may be useful:\n\n"
                                           "1. I can recognize only standart image formats: jpeg, png, bmp, etc.\n"
                                           "2. Please, don't upload too big images.\n"
                                           "3. While styling, image will be cropped (output has square form),"
                                           "so make sure that main content has center position on your image.\n"
                                           "4. Style image should has distinct texture. Good variant is some "
                                           "painting for example. You can see examples in default sets.\n"
                                           "5. I'll style any content image, but the best result I'll show "
                                           "on landscape or still life content images.\n\n"
                                           "If you are not satisfied with the result, "
                                           "you may always try again at another images:)")
    await callback_query.message.edit_reply_markup(reply_markup=back_to_menu)


# Callback for "Try on default set" button
@dp.callback_query_handler(lambda button: button.data == 'default_set')
async def set_default_set(callback_query: types.CallbackQuery):
    # Inserting of user's id to special dict for further handling default set
    if callback_query.message.chat.id not in images:
        images[callback_query.message.chat.id] = Images()
    else:
        images[callback_query.message.chat.id].content_image = 0
        images[callback_query.message.chat.id].style_image = 0
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Set default images:")
    # Random choice of bot's images and showing it to user
    default_set = await set_random_default_set()
    send_content = await bot.send_photo(callback_query.message.chat.id, default_set[0])
    content_path = await get_file_path(send_content)
    images[callback_query.message.chat.id].content_image = content_path
    send_style = await bot.send_photo(callback_query.message.chat.id, default_set[1])
    style_path = await get_file_path(send_style)
    images[callback_query.message.chat.id].style_image = style_path
    await bot.send_message(callback_query.message.chat.id,
                           "I set this default images, let's make something beautiful?",
                           reply_markup=style_button)


# Callback for "Let's get started styling!" and "Upload images" buttons
# Func for making bot able to receive images
@dp.callback_query_handler(lambda button: button.data == 'upload')
async def prepare_for_upload(callback_query: types.CallbackQuery):
    # Inserting of user's id to special dict for further handling his images
    if callback_query.message.chat.id not in images:
        images[callback_query.message.chat.id] = Images()
    else:
        images[callback_query.message.chat.id].content_image = 0
        images[callback_query.message.chat.id].style_image = 0
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("I'm ready to get your content image")
    await callback_query.message.edit_reply_markup(reply_markup=back_to_menu)


# Callback for "Discard uploaded images" button
# Func for discrading received or default images
@dp.callback_query_handler(lambda button: button.data == 'discard')
async def discard_images(callback_query: types.CallbackQuery):
    images[callback_query.message.chat.id].content_image = 0
    images[callback_query.message.chat.id].style_image = 0
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("Images discarded, let's upload something else?")
    await callback_query.message.edit_reply_markup(reply_markup=upload_button)


# Receiving of files from user
@dp.message_handler(content_types=['photo', 'document'])
async def upload_images(message: types.Message):
    try:
        # Inform user about necessity of self identification
        if message.chat.id not in images:
            await message.reply("Please, firstly push 'Upload images' to make me able to receive your images.",
                                reply_markup=upload_button)
            return
        # Get the path of received image
        image = await get_file_path(message)

        # Inform user if bot didn't recognize file format
        if not image:
            await message.reply("Unrecognized format of file. Please provide me with image.",
                                reply_markup=upload_button)
            return

        # Put the images to special dict
        if images[message.chat.id].content_image == 0:
            images[message.chat.id].content_image = image
        else:
            images[message.chat.id].style_image = image

    except Exception as ex:
        await message.reply("Something went wrong. Please, check recommendations and try later.",
                            reply_markup=menu)
        # Notification for creator about unexpected results
        await bot.send_message(config("ERROR_NOTIFICATION"), "Error occured: " + str(ex))

    # Inform user about successful getting of images
    if images[message.chat.id].content_image != 0 and images[message.chat.id].style_image == 0:
        await message.reply("Nice. Now upload style image")
    elif images[message.chat.id].content_image != 0 and images[message.chat.id].style_image != 0:
        await bot.send_message(message.chat.id,
                               "Cool! Now I'm ready to style your content image with style image.",
                               reply_markup=style_button)


# Callback for "Style it!" button
@dp.callback_query_handler(lambda button: button.data == 'style_it')
async def styling_handler(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    await callback_query.message.edit_text("I started the process. It may take few minutes.\n"
                                           "Actually, you can close the chat, "
                                           "I'll send you a message when it get ready.")
    chat_id = callback_query.message.chat.id
    content_image = images[callback_query.message.chat.id].content_image
    style_image = images[callback_query.message.chat.id].style_image
    # Define threads for simultaneous styling in case of initiating the process from different users
    styling_process = threading.Thread(
        target=lambda chat, content, style:
        asyncio.run(styling(chat_id, content_image, style_image)),
        args=(chat_id, content_image, style_image))
    styling_process.start()


# Func for calling style transferring process
async def styling(chat, content, style):
    # Define auxiliary session for async processing
    aux_bot = Bot(token=config("API_TOKEN"))
    try:
        styled = await model.transferring(content, style)
        await aux_bot.send_photo(chat, photo=styled)

    except Exception as ex:
        await aux_bot.send_message(chat, "Something went wrong. Please, try later.",
                                   reply_markup=back_to_menu)
        # Notification for creator about unexpected results
        await aux_bot.send_message(config("ERROR_NOTIFICATION"), "Error occured: " + str(ex))

    await aux_bot.send_message(chat, "That's it! Let's style something else?",
                               reply_markup=upload_button)
    await aux_bot.close()
    del images[chat]

if __name__ == '__main__':
    # executor.start_polling(dp, skip_updates=True)
    start_webhook(
        dispatcher=dp,
        skip_updates=True,
        webhook_path=wh_path,
        host=wa_host,
        port=wa_port,
        on_startup=on_startup,
        on_shutdown=on_shutdown)
